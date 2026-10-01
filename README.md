# LivDet2015 synthetic spoof fingerprint generator

Generate synthetic spoof fingerprints from bona fide images using Stable
Diffusion 1.5 Img2Img, Canny ControlNet, a shared LoRA, and a caption-anchored
residual PromptLearner for each sensor-material combination.

The PromptLearner caches the frozen CLIP caption embedding and learns a rank-8
residual in its 77-by-768 output space. Norm clipping and anchor regularization
bound the adjustment. Each combination has 6,760 trainable parameters.

## Scope

This repository contains ROI preprocessing, Training-only metadata construction,
shared LoRA training, residual prompt training, and bona fide-to-spoof generation.
It supports the four LivDet2015 sensors and 15 combinations listed in
[defaults.json](configs/livdet2015/defaults.json).

LivDet images, pretrained base models, trained project checkpoints, and generated
images are obtained or created separately. Quality scoring, PAD classifier
training/evaluation, source retrieval, and study-specific evaluation split lists
are outside this repository's scope. Metadata and generation manifests describe
the data supplied to each run; they are not archived manifests of published results.

## Environment

Use Python 3.11 on Linux or WSL2. Training uses a single CUDA GPU; the default
prompt precision requires BF16 support. Select a device with
`CUDA_VISIBLE_DEVICES`. Shell scripts use `PYTHON` when set, then the project's
`.venv/bin/python`, then `python3`.

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The CUDA wheel must be compatible with the installed NVIDIA driver and GPU.
See the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).
Direct dependency versions are pinned in `requirements.txt`; resolved versions
are saved in each run. Training scripts do not install packages automatically.

## Dataset layout

Obtain the dataset from its provider. Set `LIVDET2015_ROOT` to the directory
containing the official `Training` and `Testing` partitions. The generator accepts
both the flat layout below and the nested sensor layout
`Training/<sensor>/<sensor>/Live` or `Fake/<material>`.

```text
/path/to/livdet2015/
  Training/
    CrossMatch/
      live/
      fake/
        Body Double/
        Ecoflex/
        Playdoh/
    Digital_Persona/
    GreenBit/
    Hi_Scan/
  Testing/
    CrossMatch/
      live/
      fake/
    Digital_Persona/
    GreenBit/
    Hi_Scan/
```

Use the sensor names shown above. Authentication directory names are recognized
case-insensitively. Material names are converted to combination keys by replacing
spaces with underscores. The lexicon is
[configured here](configs/livdet2015/prompt_lexicon.json).

For raw images, extract square foreground ROIs into a separate empty directory:

```bash
python preprocess_livedet.py \
  --src_dir /path/to/raw/LivDet2015 \
  --dst_dir /path/to/roi/livdet2015
export LIVDET2015_ROOT=/path/to/roi/livdet2015
```

This preserves Training/Testing and sensor/material subdirectories and writes
PNG images, a preprocessing configuration, and a source/output hash manifest.
Source stems are retained; ambiguous stems across file extensions cause an error.
Use the same preprocessed source files when comparing runs.

Metadata construction reads only Training, excludes the top-level
`Time_Series` directory, and fails on unreadable images. Shared LoRA training
uses real Training images; each PromptLearner uses real Training spoofs of its
combination. Testing images and held-out Testing materials are retained in the
dataset. Generation targets are specified by the trained combinations rather
than inferred from Testing spoof folders.

## Base models

The configured model identifiers are:

- [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [Canny ControlNet](https://huggingface.co/lllyasviel/sd-controlnet-canny)

To use complete local diffusers-format snapshots:

```bash
export LOCAL_MODEL_DIR=/path/to/stable-diffusion-v1-5
export LOCAL_CONTROLNET_DIR=/path/to/sd-controlnet-canny
```

Local directories under `hf_models/` are also detected; see
[hf_models/README.md](hf_models/README.md). An invalid explicitly configured
local path causes an error. Without a local model, a Hub revision is resolved
and recorded before loading. Use `HF_TOKEN` for Hub authentication if needed.

## Configuration

[defaults.json](configs/livdet2015/defaults.json) is the active configuration
loaded by all four stage scripts and the full pipeline. Sensor combinations
share the same numerical settings.

| Setting | Value |
|---|---|
| Resolution | 512 x 512 |
| Shared LoRA | rank 32; 15,000 steps; learning rate 1e-4 |
| LoRA batch / accumulation / precision | 1 / 4 / FP16 |
| Residual prompt | rank 8; 20,000 steps; learning rate 1e-3 |
| Prompt batch / accumulation / precision | 4 / 2 / BF16 |
| Prompt checkpoints | every 4,000 steps, plus the final step |
| Prompt residual scale / clipping ratio | 1.0 / 0.25 |
| Anchor / residual loss weights | 1.0 / 1e-4 |
| Anchor cosine lower bound / norm ratio interval | 0.97 / [0.9, 1.1] |
| Sampling | SDE DPM-Solver++ with Karras sigmas; 30 steps |
| Guidance / Img2Img strength | 7.0 / 0.92 |
| ControlNet / LoRA scale | 0.6 / 0.8 |
| Canny thresholds | 50 / 200 |
| Base seed | 42 |

Prompt training uses AdamW with weight decay 0.01, epsilon 1e-8, Min-SNR
gamma 5, and gradient clipping at 1.0. Trainable residual factors retain FP32
master parameters under mixed-precision execution.

Pass `--config /path/to/config.json` to use another complete configuration.
Use a different artifact root or run ID when changing settings. Low-level
Python trainers also expose their arguments, while the stage scripts forward
all configured values explicitly.

## Run the pipeline

```bash
export LIVDET2015_ROOT=/path/to/roi/livdet2015
export WEIGHT_ROOT=/path/to/weights/livdet2015
export RESULT_ROOT=/path/to/results/livdet2015
export CUDA_VISIBLE_DEVICES=0

bash 01_build_metadata_livdet2015.sh
bash 02_train_lora.sh
bash 03_train_prompt_combo.sh
bash 04_generate_synthetic_spoof.sh
```

Or run the same four stages with:

```bash
bash run_pipeline.sh
```

Generation uses the shared LoRA and residual checkpoints produced in the
configured weight root. The default output includes two separate source splits:

- `Training`: bona fide Training sources transformed into synthetic spoofs,
  for training augmentation.
- `Testing`: bona fide Testing sources transformed into synthetic spoofs,
  for synthetic-attack evaluation.

Keep these populations separate in downstream experiments. This pipeline does
not train PAD classifiers or construct their evaluation protocols.

Restrict generation or select a combination:

```bash
bash 04_generate_synthetic_spoof.sh --split Training
bash 04_generate_synthetic_spoof.sh --split Testing --sensor CrossMatch
bash 03_train_prompt_combo.sh --combo CrossMatch_Body_Double
bash 04_generate_synthetic_spoof.sh --split Testing \
  --combo CrossMatch_Body_Double --max-images 10 --run-id sample
```

`--dry-run` prints planned stage commands without running them or writing
artifacts. It does not verify GPU execution or reproduce results.

## Reproducibility and resuming

Each generated image uses:

```text
seed = base_seed + Adler32(combination_key) + Adler32(source_stem)
```

Adler32 is computed over UTF-8 bytes, without a modulo reduction. Images are
saved in grayscale. The manifest records the source, split, combination, derived
seed, and source/output hashes. Configurations include model identities,
checkpoint hashes, and software versions.

For inference-only seed comparisons with fixed trained checkpoints:

```bash
bash 04_generate_synthetic_spoof.sh --split Testing --generation-seed 123 --run-id seed123
```

`--seed` changes the common base seed; `--generation-seed` changes only the
generation seed. Keep the source files, model snapshots, configuration, and
environment fixed when comparing outputs.

Repeated generation validates existing outputs against their manifest and appends
new records. A changed configuration, checkpoint, source, software version, or
modified recorded output requires a separate run directory. An unrecorded image
left by an interruption is reported explicitly. A `.run.lock` prevents concurrent
writers; remove that lock only after confirming its process has stopped.

Completed training runs are reused only after validating their configuration
and final artifact hash. To resume an interrupted run from its latest complete
training state:

```bash
bash 02_train_lora.sh --resume
bash 03_train_prompt_combo.sh --resume
```

Checkpoint state contains optimizer, scheduler, random states, and the data
cursor. With `--resume`, combinations that have not started are trained normally;
an interrupted run without a complete checkpoint is reported explicitly.
Start a new weight directory for a fresh training run. Checkpoints that
contain only model parameters cannot serve as full training resume states.
Bitwise equivalence across different GPUs, CUDA versions, or software versions
is not guaranteed.

## Outputs

```text
WEIGHT_ROOT/
  metadata/
    metadata_lora_all.jsonl
    metadata_prompt_combo.jsonl
    class_index_prompt_combo.json
    combo_manifest.json
  lora/shared/
    train_config.json
    train.log
    checkpoint-*/
    pytorch_lora_weights.safetensors
    DONE
  prompt_combo/<combination>/
    train_config.json
    train.log
    checkpoint-*/
    final_prompt_learner.pt
    DONE

RESULT_ROOT/<run-id>/<Training|Testing>/<sensor>/
  inference_config.json
  generation_plan.json
  generation_manifest.jsonl
  <combination>/.../*_gen.png
```

Training logs are written to each run's `train.log`. Metadata image paths are
relative to the Training partition. Generated records use paths relative to
`LIVDET2015_ROOT` and the corresponding output directory.

## Licensing

A license for project-authored code has not been specified. Third-party notices
and the bundled example's license are provided in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
