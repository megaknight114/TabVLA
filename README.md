# DropVLA: An Action-Level Backdoor Attack on Vision-Language-Action Models

**Accepted at IROS 2026**

Zonghuan Xu, Jiayu Li, Yunhan Zhao, Xiang Zheng, Xingjun Ma, and Yu-Gang Jiang

[[Paper](https://arxiv.org/abs/2510.10932)] [[Dataset](https://huggingface.co/datasets/Holomegaknight/openvla-oft-backdoor)] [[Citation](#citation)]

DropVLA studies action-level backdoor attacks on vision-language-action (VLA) models. Instead of forcing one fixed output, the attack implants a reusable malicious action primitive, such as opening the gripper, that can be activated at attacker-chosen decision points while the policy continues the surrounding task.

This repository provides the data-poisoning, OpenVLA-OFT fine-tuning, and LIBERO evaluation code used in the paper.

## Highlights

- **Action-level attack:** implants a composable malicious action rather than a fixed end-to-end behavior.
- **Pipeline-black-box setting:** requires no access to model parameters or the training process and uses only a small poisoned data subset.
- **Strong attack efficacy:** on OpenVLA-7B and LIBERO, vision-only DropVLA reaches 98.67--99.83% attack success with 0.31% poisoned episodes while retaining 98.50--99.17% clean-task success.
- **Transfer and physical evaluation:** the attack transfers across LIBERO task suites and is further evaluated on a 7-DoF Franka robot.

## Installation

The released scripts target Python 3.9 and a CUDA-enabled PyTorch environment. Install the PyTorch build appropriate for your system from the [official instructions](https://pytorch.org/get-started/locally/), then set up the repository:

```bash
git clone https://github.com/megaknight114/DropVLA.git
cd DropVLA

conda create -n dropvla python=3.9 -y
conda activate dropvla

# Install PyTorch first, using the command appropriate for your CUDA version.
pip install torch torchvision torchaudio

# Core model and experiment dependencies.
pip install "transformers==4.54.1" "peft==0.16.0" "tokenizers==0.21.4"
pip install accelerate "bitsandbytes==0.46.1" draccus wandb huggingface_hub
pip install tensorflow tensorflow-datasets pillow tqdm
```

Install LIBERO next to this repository:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ../LIBERO
pip install -e ../LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt

export DROPVLA_ROOT="$(pwd)"
export LIBERO_PATH="$(cd ../LIBERO && pwd)"
export DATA_DIR="$DROPVLA_ROOT/datasets"
export RUN_DIR="$DROPVLA_ROOT/runs"
export PYTHONPATH="$DROPVLA_ROOT:$LIBERO_PATH:$PYTHONPATH"
```

Run the commands below from the DropVLA repository root. `flash-attn` and `deepspeed` are optional and are not required for the provided single-GPU examples.

## Data and Checkpoints

The released poisoned datasets are available on [Hugging Face](https://huggingface.co/datasets/Holomegaknight/openvla-oft-backdoor). Place the downloaded RLDS datasets under `$DATA_DIR`, or pass their location directly with `--data_root_dir`.

For fine-tuning, start from [`openvla/openvla-7b`](https://huggingface.co/openvla/openvla-7b) or a compatible local checkpoint. For evaluation, provide the resulting checkpoint directory through `--pretrained_checkpoint`.

## Build a Backdoor Dataset

The poisoning workflow is:

1. Convert a LIBERO RLDS dataset to the readable episode format.
2. Inject a visual, language, or joint trigger.
3. Convert the poisoned episodes back to RLDS.

### 1. Convert RLDS to readable episodes

```bash
python rlds_to_readable.py \
  --input_dir <TFRECORD_VERSION_DIR> \
  --output_dir <READABLE_DIR>
```

### 2. Inject the trigger

Vision-only example using the paper's 0.31% episode-poisoning rate:

```bash
python visual_backdoor_attack.py \
  --dataset_path <READABLE_DIR> \
  --episode_ratio 0.0031 \
  --step_ratio 1 \
  --output_name <POISONED_DATASET_NAME> \
  --language_suffix ""
```

For a language-only trigger, disable image modification and supply a suffix:

```bash
python visual_backdoor_attack.py \
  --dataset_path <READABLE_DIR> \
  --episode_ratio 0.0031 \
  --step_ratio 1 \
  --output_name <POISONED_DATASET_NAME> \
  --disable_image_backdoor \
  --language_suffix "carefully"
```

For a joint trigger, leave image modification enabled and provide a non-empty language suffix. The script copies the readable dataset before modifying it.

### 3. Convert readable episodes back to RLDS

```bash
python readable_to_rlds.py \
  --readable_dir <POISONED_READABLE_DIR> \
  --output_dir <RLDS_OUTPUT_DIR> \
  --dataset_name <DATASET_NAME>
```

## Fine-Tuning

The following command shows the main OpenVLA-OFT configuration used by the released training pipeline. Adjust paths, dataset name, batch size, and logging settings for your hardware.

```bash
accelerate launch vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir "$DATA_DIR" \
  --dataset_name <DATASET_NAME> \
  --run_root_dir "$RUN_DIR" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 3e-4 \
  --num_steps_before_decay 10000 \
  --max_steps 15005 \
  --save_freq 3000 \
  --save_latest_checkpoint_only True \
  --image_aug True \
  --lora_rank 32 \
  --seed 42
```

To run without `accelerate`, replace `accelerate launch` with `python`. If `bitsandbytes` is unavailable, disable quantization in the model configuration.

## Evaluation

The evaluation script supports clean, language-triggered, and visual-triggered rollouts. Explicitly set both trigger flags so the evaluation condition is unambiguous.

### Clean evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$RUN_DIR/<CHECKPOINT>" \
  --task_suite_name libero_spatial \
  --use_backdoor_instruction False \
  --use_visual_backdoor False
```

### Language-triggered evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$RUN_DIR/<CHECKPOINT>" \
  --task_suite_name libero_spatial \
  --use_backdoor_instruction True \
  --language_suffix "carefully" \
  --use_visual_backdoor False
```

### Visual-triggered evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$RUN_DIR/<CHECKPOINT>" \
  --task_suite_name libero_spatial \
  --use_backdoor_instruction False \
  --use_visual_backdoor True \
  --backdoor_activation_height_m 0.06
```

For joint evaluation, enable both trigger flags. Other LIBERO suites can be selected with `--task_suite_name`.

## Reproducibility Notes

- Record the repository commit, dataset snapshot, checkpoint step, and random seed for every run.
- Keep clean and poisoned datasets in separate directories.
- Log trigger modality, poisoning ratios, language suffix, and visual activation threshold with each result.
- Report clean task success together with attack success.

## Repository Structure

```text
DropVLA/
├── visual_backdoor_attack.py
├── rlds_to_readable.py
├── readable_to_rlds.py
├── vla-scripts/finetune.py
├── experiments/robot/libero/run_libero_eval.py
├── openvla/
├── prismatic/
└── rlds_dataset_builder/
```

## Citation

If you find the paper, data, or code useful, please cite:

```bibtex
@misc{xu2026dropvla,
  title         = {DropVLA: An Action-Level Backdoor Attack on Vision-Language-Action Models},
  author        = {Zonghuan Xu and Jiayu Li and Yunhan Zhao and Xiang Zheng and Xingjun Ma and Yu-Gang Jiang},
  year          = {2026},
  eprint        = {2510.10932},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url           = {https://arxiv.org/abs/2510.10932},
  note          = {Accepted at IROS 2026}
}
```

The proceedings citation will be added once the official IROS 2026 record is available.

## License

This repository is released under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](LICENSE).
