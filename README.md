# TabVLA

> Reproduce / study **with/without backdoor** training & evaluation of Vision-Language-Action models on **LIBERO**.Supports building language / visual / joint (vision+language) backdoor datasets, plus example evaluation & finetune scripts.

> **Optional deps (READ ME)**
> 
> * `flash-attn`, `accelerate`, `deepspeed`, `bitsandbytes` are **all optional**.
> * **Recommended for reproducibility:** install **`accelerate`** and **`bitsandbytes`** (we default to these in examples).
> * If you skip either one, see **§6 No-Quant / No-Accelerate** to adjust launch flags or model construction.

* * *

## Table of Contents

* [0) Conventions & ENV VARS](#0-conventions--env-vars)
* [1) Quickstart](#1-quickstart)
  * [1.1 Create env & install core deps](#11-create-env--install-core-deps)
  * [1.2 Optional extras (install if you like)](#12-optional-extras-install-if-you-like)
  * [1.3 Install LIBERO](#13-install-libero)
  * [1.4 Data & checkpoints](#14-data--checkpoints)
  * [1.5 Minimal evaluation (single GPU)](#15-minimal-evaluation-single-gpu)
* [2) Full Installation (layered & optional)](#2-full-installation-layered--optional)
* [3) Evaluate](#3-evaluate)
* [4) Build Backdoor Datasets](#4-build-backdoor-datasets)
* [5) Finetune](#5-finetune)
* [6) No-Quant / No-Accelerate](#6-noquant--noaccelerate)
* [7) Reproducibility](#7-reproducibility)
* [8) Troubleshooting / FAQ](#8-troubleshooting--faq)
* [9) Repo Layout & Scripts](#9-repo-layout--scripts)
* [10) Citation](#10-citation)
* [11) License](#11-license)

* * *

## 0) Conventions & ENV VARS

To avoid long paths and reduce path mistakes, set:

    export ENV_NAME=openvla-oft
    export ROOT=$HOME/openvla-oft
    export DATA_DIR=$ROOT/datasets/openvla
    export RUN_DIR=$ROOT/RUN
    export LIBERO_PATH=$ROOT/LIBERO

* * *

## 1) Quickstart

> To avoid duplication, **finetune commands live only in §5**. Quickstart shows install + a minimal **evaluate**.When you're ready to train, jump to **§5 Finetune**.

### 1.1 Create env & install core deps

    # Conda env
    conda create -n $ENV_NAME python=3.9 -y
    conda activate $ENV_NAME
    
    # PyTorch (pick the right command for your system: https://pytorch.org/get-started/locally/)
    pip install torch torchvision torchaudio
    
    # Clone & editable install
    git clone https://github.com/moojink/openvla-oft.git $ROOT
    cd $ROOT
    pip install -e .
    
    # --- Core pinned deps (install directly; no requirements.txt) ---
    pip install "transformers==4.54.1" "peft==0.16.0" "tokenizers==0.21.4"

### 1.2 Optional extras (install if you like)

    # Recommended (we assume these in examples; helps reproducibility):
    pip install accelerate
    pip install "bitsandbytes==0.46.1"
    
    # Fully optional accelerators (ok to skip if they fail on your system):
    pip install ninja packaging
    pip install "flash-attn==2.5.5" deepspeed

### 1.3 Install LIBERO

    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git $LIBERO_PATH
    pip install -e $LIBERO_PATH
    pip install -r experiments/robot/libero/libero_requirements.txt

### 1.4 Data & checkpoints

* Dataset (Hugging Face): https://huggingface.co/datasets/Holomegaknight/openvla-oft-backdoor/tree/mainDownload to: `$DATA_DIR/modified_libero_rlds`
* Checkpoints: use your own, or `openvla/openvla-7b`, or any checkpoint under `$RUN_DIR`.

Expected structure (example):

    $ROOT/
      RUN/
      LIBERO/
      datasets/openvla/
        modified_libero_rlds/
          libero_spatial_no_noops_...

### 1.5 Minimal evaluation (single GPU)

    # Make LIBERO visible
    export PYTHONPATH=$LIBERO_PATH:$PYTHONPATH
    
    CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint $RUN_DIR/vl5p00 \
      --task_suite_name libero_spatial

> Ready to finetune? Jump to **§5 Finetune**.

* * *

## 2) Full Installation (layered & optional)

* **Core (pinned):** `transformers==4.54.1`, `peft==0.16.0`, `tokenizers==0.21.4`
* **Nice to have (recommended for reproducibility):**
  * `accelerate` – convenient multi-process launcher
  * `bitsandbytes` – 4-bit quantization
* **Fully optional:**
  * `flash-attn==2.5.5` – speedups (skip if it fails)
  * `deepspeed` – large-scale training

* * *

## 3) Evaluate

### 3.1 Single-GPU (baseline)

    export PYTHONPATH=$LIBERO_PATH:$PYTHONPATH
    
    CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint $RUN_DIR/vl5p00 \
      --task_suite_name libero_spatial

### 3.2 Batch evaluate multiple settings (tmux, optional)

Create `scripts/eval_libero_tmux.sh`:

    #!/usr/bin/env bash
    set -e
    
    SESSION="libero_eval_0_3"
    ENV_NAME="${ENV_NAME:-openvla-oft}"
    CHECKPOINT="${RUN_DIR:-$HOME/openvla-oft/RUN}/vl5p00"
    TASK_SUITE="libero_spatial"
    LIBERO_PATH="${LIBERO_PATH:-$HOME/openvla-oft/LIBERO}"
    PYFILE="experiments/robot/libero/run_libero_eval.py"
    
    # heights: first clean (0.0), others with backdoor example heights
    heights=(0.0 0.06 0.07 0.08)
    flags=("" "--use_visual_backdoor True")
    
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      tmux attach -t "$SESSION"; exit 0
    fi
    
    tmux new-session -d -s "$SESSION" -n main
    tmux split-window -h -t "$SESSION":0
    tmux split-window -v -t "$SESSION":0.0
    tmux split-window -v -t "$SESSION":0.1
    tmux select-layout -t "$SESSION":0 tiled
    
    for i in {0..3}; do
      f=${flags[$((i>0))]}
      h=${heights[$i]}
      CMD="bash -i -c 'conda activate ${ENV_NAME} && export PYTHONPATH=${LIBERO_PATH}:\$PYTHONPATH && CUDA_VISIBLE_DEVICES=${i} python ${PYFILE} --pretrained_checkpoint ${CHECKPOINT} --task_suite_name ${TASK_SUITE} ${f} --backdoor_activation_height_m ${h}'"
      echo "[Pane $i | GPU $i | height=$h | flag=$f]"
      echo "  $CMD"
      tmux send-keys -t "$SESSION":0.$i "$CMD" Enter
    done
    
    tmux attach -t "$SESSION"

Run:

    bash scripts/eval_libero_tmux.sh

* * *

## 4) Build Backdoor Datasets

### 4.1 RLDS ↔ readable format conversion

    # RLDS -> human-readable
    python $ROOT/rlds_to_readable.py
    
    # readable -> RLDS
    python $ROOT/readable_to_rlds.py

### 4.2 Inject backdoors on readable data (auto backs-up the clean set)

Naming suggestion: `{domain}_{mark}_{type+ratio}{text_suffix}`

* Language-only (`l`): `..._l0p31carefully`
* Vision-only (`v`): `..._v0p31` (leave language suffix empty)
* Joint (`vl`): `..._vl0p31sentence`

Examples:

    # Language-only backdoor
    python visual_backdoor_attack.py \
      --dataset_path $DATA_DIR/readable_dataset/libero_spatial_no_noops_readable \
      --episode_ratio 0.05 \
      --step_ratio 1 \
      --output_name libero_spatial_no_noops_readable_l0p31carefully \
      --disable_image_backdoor \
      --language_suffix "carefully"
    
    # Vision-only backdoor
    python visual_backdoor_attack.py \
      --dataset_path $DATA_DIR/readable_dataset/libero_spatial_no_noops_readable \
      --episode_ratio 0.05 \
      --step_ratio 1 \
      --output_name libero_spatial_no_noops_readable_v0p31 \
      --language_suffix ""
    
    # Joint vision+language backdoor
    python visual_backdoor_attack.py \
      --dataset_path $DATA_DIR/readable_dataset/libero_spatial_no_noops_readable \
      --episode_ratio 0.05 \
      --step_ratio 1 \
      --output_name libero_spatial_no_noops_readable_vl0p31sentence \
      --language_suffix "After lifting the object, when a red dot appears in the top-left corner, release the gripper to drop the object."

> After injection, re-build RLDS if needed (see 4.1).

* * *

## 5) Finetune

> Uses `accelerate` by default (recommended). If you don't want `accelerate`, see **§6.2**.

### 5.1 Full

    accelerate launch vla-scripts/finetune.py \
      --vla_path $RUN_DIR/openvla-7b \
      --data_root_dir $DATA_DIR/modified_libero_rlds \
      --dataset_name libero_spatial_no_noops_vl5p00 \
      --run_root_dir $RUN_DIR \
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
      --wandb_entity "" \
      --wandb_project "" \
      --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state \
      --seed 42

###

* * *

## 6) No-Quant / No-Accelerate

### 6.1 Disable quantization (run without bitsandbytes)

In model construction (e.g., `finetune.py` / `model_init.py`), ensure that when disabling quantization:

* Set `quantization_config=None`

### 6.2 Run without accelerate

Replace:

    accelerate launch vla-scripts/finetune.py ...

with:

    python vla-scripts/finetune.py ...

For multi-GPU/distributed/mixed precision, configure your own `torchrun` or `deepspeed` launch (see **§5.2–§5.3**).

* * *

## 7) Reproducibility

* **Pin versions:** install core packages via the commands in **§1.1**.
* **Recommend:** install **`accelerate`** and **`bitsandbytes`** (our examples assume them; many configs/logs depend on these).
* **Seeds:** fix `--seed` in training/eval scripts; log all toggles (quantization, flash-attn, deepspeed).
* **Data immutability:** keep a copy of the exact dataset snapshot used; record dataset names (e.g., `libero_spatial_no_noops_vl5p00`).
* **Checkpoints:** record commit hash and checkpoint step; prefer `--save_latest_checkpoint_only` to limit disk.
* **WandB / Logs:** store hyperparameters, env info, and git SHA for each run.

* * *

## 8) Troubleshooting / FAQ

* **`flash-attn` fails to build**Skip it; it's not required. You still can finetune/eval.
  
* **`bitsandbytes` CUDA mismatch**Check GPU driver/CUDA; recent 12.x works well. If stuck, use **§6.1** to disable quantization.
  
* **No `accelerate` installed**Use `python ...` (see **§6.2**), or `torchrun`/`deepspeed`.
  
* **LIBERO import issues**Ensure `export PYTHONPATH=$LIBERO_PATH:$PYTHONPATH`.
  

* * *

## 9) Repo Layout & Scripts

    openvla-oft/
      experiments/robot/libero/run_libero_eval.py
      vla-scripts/finetune.py
      datasets/openvla/rlds_to_readable
      rlds_dataset_builder/
        libero_spacial/
          libero_spacial_dataset_builder.py
      scripts/
        eval_libero_tmux.sh   # optional, multi-setting evaluation

* * *

## 10) Citation

If you find this repository, data, or scripts useful for your research, please cite it as:

    @misc{tabvla_2025,
      title  = {TabVLA: Targeted Backdoor Attacks on Vision-Language-Action Models},
      author = {Anonymous Authors},
      year   = {2025},
      note   = {Under review},
      url    = {https://github.com/megaknight114/TabVLA}
    }

*This citation entry will be updated with the final author list and venue after publication.*

---

## 11) License

This repository is licensed under the  
**Creative Commons Attribution–NonCommercial–NoDerivatives (CC BY-NC-ND 4.0)** License.

You are free to share this work (copy and redistribute the material in any medium or format) under the following terms:

- **Attribution** — You must give appropriate credit.  
- **NonCommercial** — You may not use the material for commercial purposes.  
- **NoDerivatives** — If you remix, transform, or build upon the material, you may not distribute the modified material.

Full license text is provided in the `LICENSE` file.
