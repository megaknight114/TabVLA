#!/usr/bin/env python3
"""
Activation map visualization for OpenVLA vision tokens.

This script forwards an image/video frame through the OpenVLA model and
extracts the projected patch embeddings (vision tokens) to build a spatial
activation heatmap aligned to the model's preprocessed image. The heatmap is
overlaid onto the image and saved to disk.

Examples:
  - Image:
      python scripts/visualize_openvla_activations.py \
        --checkpoint /home/xuzonghuan/openvla-oft/RUN/openvla-7b \
        --image /path/to/image.jpg \
        --instruction "pick up the black bowl in the top drawer"

  - Video:
      python scripts/visualize_openvla_activations.py \
        --checkpoint /home/xuzonghuan/openvla-oft/RUN/openvla-7b \
        --video /path/to/video.mp4 \
        --instruction "pick up the black bowl in the top drawer" \
        --stride 5
"""

import argparse
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from PIL import Image

from experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    prepare_images_for_vla,
)


def print_gpu_info():
    """Print GPU information for debugging"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    else:
        print("No CUDA GPUs available")


def build_cfg(args: argparse.Namespace) -> SimpleNamespace:
    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.checkpoint,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        num_images_in_input=1,
        center_crop=args.center_crop,
        use_film=False,
        use_proprio=False,
        unnorm_key=None,
        lora_rank=32,
        use_l1_regression=False,
        use_diffusion=False,
        num_diffusion_steps_train=1000,
        num_diffusion_steps_inference=50,
        # Force float32 to avoid dtype mismatches during visualization
        force_float32=True,
        # Multi-GPU support
        device_map=getattr(args, 'device_map', None),
        max_memory=getattr(args, 'max_memory', None),
    )
    return cfg


def _get_vision_backbone(model: torch.nn.Module):
    if hasattr(model, "vision_backbone"):
        return getattr(model, "vision_backbone")
    if hasattr(model, "model") and hasattr(model.model, "vision_backbone"):
        return model.model.vision_backbone
    if hasattr(model, "base_model") and hasattr(model.base_model, "vision_backbone"):
        return model.base_model.vision_backbone
    return None


def _get_patch_grid_size(model: torch.nn.Module, num_images_in_input: int) -> Tuple[int, int]:
    vb = _get_vision_backbone(model)
    # Use SigLIP featurizer for grid size
    if vb is not None and hasattr(vb, "featurizer") and hasattr(vb.featurizer, "patch_embed"):
        pe = vb.featurizer.patch_embed
        if hasattr(pe, "grid_size"):
            grid = pe.grid_size
            if isinstance(grid, (list, tuple)) and len(grid) == 2:
                return int(grid[0]), int(grid[1])
    # Fallback: assume square grid from token count per image
    # Compute tokens per image by dividing by number of images (fused backbone doesn't change patch count)
    # This requires knowing the token count; we will infer at runtime based on projector_features.shape if needed.
    raise RuntimeError("Could not determine ViT patch grid size from model featurizer")


def _activation_from_projector_features(projector_features: torch.Tensor, grid_hw: Tuple[int, int]) -> np.ndarray:
    # projector_features: [B, P, D]; take norm over D to get scalar per patch
    with torch.no_grad():
        feat = projector_features.float().abs()
        # L2 norm per patch
        patch_scores = torch.linalg.vector_norm(feat, ord=2, dim=-1)  # [B, P]
        patch_scores = patch_scores[0]  # [P]
        h, w = grid_hw
        if patch_scores.numel() != h * w:
            # If multiple images in input were used, tokens are concatenated along patch dimension.
            # Here we only support the single-image case in this script.
            raise ValueError(f"Unexpected token count {patch_scores.numel()} for grid {h}x{w}")
        act = patch_scores.view(h, w).cpu().numpy()
        # Normalize to 0..1
        act = act - act.min()
        maxv = act.max() if act.max() > 0 else 1.0
        act = act / maxv
        return act


def _overlay_heatmap_on_image(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    assert 0.0 <= alpha <= 1.0
    img_rgb = np.array(pil_image)
    if cv2 is None:
        # Simple RGB blend without colormap
        heat_rgb = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        heat_rgb = np.stack([heat_rgb] * 3, axis=-1)
        overlay = (alpha * heat_rgb + (1 - alpha) * img_rgb).astype(np.uint8)
        return Image.fromarray(overlay)

    h, w = img_rgb.shape[:2]
    heat_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_uint8 = (np.clip(heat_resized, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    overlay = (alpha * color + (1 - alpha) * img_rgb).astype(np.uint8)
    return Image.fromarray(overlay[:, :, ::-1][:, :, ::-1]) if False else Image.fromarray(overlay)


def _prepare_pil_for_model(pil: Image.Image, cfg: SimpleNamespace) -> Image.Image:
    # Use repo's preprocessing (resize to 224 and optional center crop) for distribution alignment
    np_img = np.array(pil.convert("RGB"))
    processed = prepare_images_for_vla([np_img], cfg)[0]
    return processed


def _forward_and_get_projector_features(
    vla: torch.nn.Module,
    processor: Any,
    pil_image: Image.Image,
    instruction: str,
) -> torch.Tensor:
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    try:
        inputs = processor(prompt, pil_image)
        
        # Move inputs to device and ensure dtype compatibility
        for k, v in inputs.items():
            if hasattr(v, "to"):
                inputs[k] = v.to(vla.device)
                # Ensure pixel_values has the right dtype to match model weights
                if k == "pixel_values":
                    # Try to get vision backbone from different possible locations
                    vision_backbone = _get_vision_backbone(vla)
                    if vision_backbone is not None:
                        try:
                            vision_params = next(iter(vision_backbone.parameters()))
                            if vision_params.dtype in [torch.bfloat16, torch.float16]:
                                inputs[k] = inputs[k].to(vision_params.dtype)
                        except StopIteration:
                            # No parameters found, keep original dtype
                            pass
        
        with torch.no_grad():
            out = vla(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                labels=None,  # Explicitly set labels to None for inference
            )
        assert hasattr(out, "projector_features") and out.projector_features is not None, "No projector_features returned"
        return out.projector_features
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        print(f"Input shapes: input_ids={inputs['input_ids'].shape if 'input_ids' in inputs else 'None'}, "
              f"pixel_values={inputs['pixel_values'].shape if 'pixel_values' in inputs else 'None'}")
        if 'inputs' in locals():
            print(f"Input dtypes: input_ids={inputs['input_ids'].dtype if 'input_ids' in inputs else 'None'}, "
                  f"pixel_values={inputs['pixel_values'].dtype if 'pixel_values' in inputs else 'None'}")
        raise


def run_on_image(args: argparse.Namespace, cfg: SimpleNamespace) -> None:
    assert os.path.isfile(args.image), f"Image not found: {args.image}"
    vla = get_vla(cfg)
    
    # Convert model to float32 to avoid dtype mismatches during visualization
    if not (args.load_in_8bit or args.load_in_4bit):
        vla = vla.float()
    
    # Apply multi-GPU setup if specified
    if hasattr(args, 'device_map') and args.device_map:
        print(f"Using device map: {args.device_map}")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
    
    processor = get_processor(cfg)

    pil_raw = Image.open(args.image).convert("RGB")
    pil_proc = _prepare_pil_for_model(pil_raw, cfg)

    # Determine grid size from model
    try:
        grid_hw = _get_patch_grid_size(vla, num_images_in_input=1)
    except Exception:
        # Fallback: infer from features
        proj = _forward_and_get_projector_features(vla, processor, pil_proc, args.instruction)
        p = proj.shape[1]
        s = int(round(p ** 0.5))
        grid_hw = (s, s)
    # Forward for features
    proj = _forward_and_get_projector_features(vla, processor, pil_proc, args.instruction)
    heat = _activation_from_projector_features(proj, grid_hw)
    vis = _overlay_heatmap_on_image(pil_proc, heat, alpha=args.alpha)

    out_path = args.output or os.path.splitext(args.image)[0] + "__openvla_act.jpg"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vis.save(out_path)
    print(f"Saved activation visualization to: {out_path}")


def run_on_video(args: argparse.Namespace, cfg: SimpleNamespace) -> None:
    assert cv2 is not None, "OpenCV (cv2) is required for video visualization. Please install opencv-python."
    assert os.path.isfile(args.video), f"Video not found: {args.video}"

    vla = get_vla(cfg)
    
    # Convert model to float32 to avoid dtype mismatches during visualization
    if not (args.load_in_8bit or args.load_in_4bit):
        vla = vla.float()
    
    # Apply multi-GPU setup if specified
    if hasattr(args, 'device_map') and args.device_map:
        print(f"Using device map: {args.device_map}")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
    
    processor = get_processor(cfg)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Failed to open video: {args.video}"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 30.0
    stride = max(1, int(args.stride))

    out_path = args.output or os.path.splitext(args.video)[0] + "__openvla_act.mp4"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    # Prioritize MP4 codecs for better compatibility
    mp4_codecs = ['mp4v', 'H264', 'X264', 'avc1', 'FMP4']
    avi_codecs = ['XVID', 'MJPG', 'DIV3', 'DIVX', 'I420']
    
    out_fps = max(1.0, fps / float(stride))
    writer = None
    original_out_path = out_path
    
    # Determine format preference based on user input
    user_specified_output = args.output is not None
    original_ext = os.path.splitext(original_out_path)[1].lower()
    
    if user_specified_output and original_ext == '.mp4':
        # User wants MP4, prioritize MP4 codecs
        codec_format_options = [(c, '.mp4') for c in mp4_codecs] + [(c, '.avi') for c in avi_codecs]
    elif user_specified_output and original_ext == '.avi':
        # User wants AVI, prioritize AVI codecs
        codec_format_options = [(c, '.avi') for c in avi_codecs] + [(c, '.mp4') for c in mp4_codecs]
    else:
        # Default: prioritize MP4 format
        codec_format_options = [(c, '.mp4') for c in mp4_codecs] + [(c, '.avi') for c in avi_codecs]
    
    # Try codecs with user's preferred format first
    for codec, preferred_ext in codec_format_options:
        try:
            # If user specified output, try to honor their format preference
            if user_specified_output and original_ext in ['.mp4', '.avi']:
                test_path = original_out_path  # Use exact user path
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(test_path, fourcc, out_fps, (width, height))
                
                if writer and writer.isOpened():
                    out_path = test_path
                    print(f"Successfully using codec: {codec} with user-specified format: {original_ext}")
                    break
                
                if writer:
                    writer.release()
                    writer = None
            
            # Fallback: try with codec's preferred format
            fallback_path = os.path.splitext(original_out_path)[0] + preferred_ext
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(fallback_path, fourcc, out_fps, (width, height))
            
            if writer and writer.isOpened():
                out_path = fallback_path
                if user_specified_output and preferred_ext != original_ext:
                    print(f"Using codec: {codec} with format: {preferred_ext} (fallback from user-specified {original_ext})")
                else:
                    print(f"Successfully using codec: {codec} with format: {preferred_ext}")
                break
            
            if writer:
                writer.release()
                writer = None
                
        except Exception as e:
            print(f"Failed to initialize codec {codec}: {e}")
            if writer:
                writer.release()
                writer = None
            continue
    
    if not writer or not writer.isOpened():
        # Final fallback: save as individual images if video writing fails
        print("WARNING: All video codecs failed. Saving as individual frame images instead.")
        import tempfile
        frames_dir = tempfile.mkdtemp(prefix="openvla_frames_")
        print(f"Frames will be saved to: {frames_dir}")
        
        # Create a simple image sequence saver
        class ImageSequenceWriter:
            def __init__(self, output_dir, width, height):
                self.output_dir = output_dir
                self.frame_count = 0
                self.width = width
                self.height = height
                
            def write(self, frame):
                from PIL import Image
                # Convert BGR to RGB for PIL
                frame_rgb = frame[:, :, ::-1]
                img = Image.fromarray(frame_rgb)
                frame_path = os.path.join(self.output_dir, f"frame_{self.frame_count:06d}.jpg")
                img.save(frame_path, quality=95)
                self.frame_count += 1
                
            def release(self):
                print(f"Saved {self.frame_count} frames to {self.output_dir}")
                print("To create a video from frames, use:")
                print(f"ffmpeg -framerate {out_fps} -i {self.output_dir}/frame_%06d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p {original_out_path}")
                
            def isOpened(self):
                return True
        
        writer = ImageSequenceWriter(frames_dir, width, height)
        out_path = frames_dir

    # Attempt to get grid size via model; else infer on first processed frame
    grid_hw: Optional[Tuple[int, int]] = None

    idx = 0
    written = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        idx += 1

        pil_raw = Image.fromarray(frame_bgr[:, :, ::-1]).convert("RGB")
        pil_proc = _prepare_pil_for_model(pil_raw, cfg)

        proj = _forward_and_get_projector_features(vla, processor, pil_proc, args.instruction)
        if grid_hw is None:
            try:
                grid_hw = _get_patch_grid_size(vla, num_images_in_input=1)
            except Exception:
                p = proj.shape[1]
                s = int(round(p ** 0.5))
                grid_hw = (s, s)

        heat = _activation_from_projector_features(proj, grid_hw)
        # Overlay heatmap on the processed image to avoid misalignment
        vis_pil = _overlay_heatmap_on_image(pil_proc, heat, alpha=args.alpha)
        # Resize back to original frame size for writing
        vis_frame = vis_pil.resize((width, height))
        writer.write(np.array(vis_frame)[:, :, ::-1])
        written += 1

    writer.release()
    cap.release()
    
    # Verify the output
    if hasattr(writer, '__class__') and writer.__class__.__name__ == 'ImageSequenceWriter':
        # Image sequence case
        print(f"Saved activation visualization as image sequence: {out_path} ({written} frames)")
    elif os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"Saved activation visualization to: {out_path} ({written} frames)")
        
        # Try to verify the video can be opened
        test_cap = cv2.VideoCapture(out_path)
        if test_cap.isOpened():
            frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = test_cap.get(cv2.CAP_PROP_FPS)
            print(f"Video verification: {frame_count} frames at {video_fps:.2f} FPS")
            test_cap.release()
        else:
            print(f"WARNING: Generated video file exists but cannot be opened by OpenCV")
            print(f"File size: {os.path.getsize(out_path)} bytes")
            print("Try using a different video player or convert with ffmpeg:")
            print(f"ffmpeg -i {out_path} -c:v libx264 -crf 23 -preset medium {out_path.replace('.mp4', '_converted.mp4')}")
    else:
        print(f"ERROR: Failed to create video file or file is empty")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize OpenVLA activation heatmaps (vision tokens)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to local OpenVLA checkpoint directory")
    parser.add_argument("--instruction", type=str, required=True, help="Language instruction for the task")
    parser.add_argument("--image", type=str, default=None, help="Path to an input image (RGB)")
    parser.add_argument("--video", type=str, default=None, help="Path to an input video (mp4)")
    parser.add_argument("--output", type=str, default=None, help="Output path (jpg/mp4). Defaults next to input")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay alpha in [0,1]")
    parser.add_argument("--stride", type=int, default=5, help="Process every Nth frame in video mode")
    parser.add_argument("--center-crop", dest="center_crop", action="store_true", help="Apply center crop (recommended)")
    parser.add_argument("--no-center-crop", dest="center_crop", action="store_false", help="Disable center crop")
    parser.set_defaults(center_crop=True)
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit quantized mode")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit quantized mode")
    
    # Multi-GPU support arguments
    parser.add_argument("--device-map", type=str, default="auto", 
                       help="Device map for multi-GPU: 'auto', 'balanced', 'balanced_low_0', or custom dict")
    parser.add_argument("--max-memory", type=str, default=None,
                       help="Max memory per GPU (e.g., '10GB' or '{0: \"10GB\", 1: \"15GB\"}')")
    parser.add_argument("--single-gpu", action="store_true", 
                       help="Force single GPU usage (disable multi-GPU)")
    
    args = parser.parse_args()
    if (args.image is None) == (args.video is None):
        parser.error("Please specify exactly one of --image or --video")
    if not (0.0 <= args.alpha <= 1.0):
        parser.error("--alpha must be in [0,1]")
    
    # Handle multi-GPU settings
    if args.single_gpu:
        args.device_map = None
        args.max_memory = None
    elif args.device_map and args.device_map != "auto":
        # Parse custom device map if provided
        try:
            if args.device_map.startswith('{'):
                import ast
                args.device_map = ast.literal_eval(args.device_map)
        except:
            parser.error("Invalid device_map format. Use 'auto', 'balanced', or a valid dict string")
    
    # Parse max_memory if provided
    if args.max_memory:
        try:
            if args.max_memory.startswith('{'):
                import ast
                args.max_memory = ast.literal_eval(args.max_memory)
        except:
            parser.error("Invalid max_memory format. Use '10GB' or a valid dict string")
    
    return args


def main() -> None:
    args = parse_args()
    
    # Print GPU information
    print_gpu_info()
    
    # Print device configuration
    if hasattr(args, 'device_map') and args.device_map:
        print(f"\nDevice map: {args.device_map}")
    if hasattr(args, 'max_memory') and args.max_memory:
        print(f"Max memory: {args.max_memory}")
    
    cfg = build_cfg(args)
    if args.image:
        run_on_image(args, cfg)
    else:
        run_on_video(args, cfg)


if __name__ == "__main__":
    main()

