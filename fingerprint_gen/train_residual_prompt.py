#!/usr/bin/env python
# coding=utf-8
import argparse
import json
import logging
import math
import os
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader

from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from prompt_learner import PromptLearner
from sd15.utils.run_logger import save_json

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Caption-anchored residual PromptLearner with frozen SD1.5 + LoRA."
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--checkpointing_steps", type=int, default=4000)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Resume full training state from a checkpoint directory or "latest".',
    )
    parser.add_argument("--customize_prefix", type=str, default="")
    parser.add_argument("--customize_suffix", type=str, default="")
    parser.add_argument("--residual_rank", type=int, default=8)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--residual_kappa", type=float, default=0.25)
    parser.add_argument("--anchor_loss_weight", type=float, default=1.0)
    parser.add_argument("--residual_loss_weight", type=float, default=1e-4)
    parser.add_argument("--cosine_min", type=float, default=0.97)
    parser.add_argument("--norm_ratio_min", type=float, default=0.9)
    parser.add_argument("--norm_ratio_max", type=float, default=1.1)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--lora_scale", type=float, default=0.8)
    parser.add_argument(
        "--metadata_jsonl",
        type=str,
        default=None,
        help="Enable metadata training by aligning file_name and class_id.",
    )
    parser.add_argument(
        "--class_index_json",
        type=str,
        default=None,
        help="Class index containing num_classes, id_to_style_key, and style_key_to_id.",
    )
    parser.add_argument(
        "--n_cls",
        type=int,
        default=None,
        help="Class count; inferred from class_index_json when using metadata.",
    )
    args = parser.parse_args()
    return args


def load_metadata_pairs(metadata_jsonl: str, train_data_dir: str):
    rows = []
    with open(metadata_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["file_name"])
    pairs = []
    for r in rows:
        rel = r["file_name"]
        cid = int(r["class_id"])
        full = os.path.join(train_data_dir, rel.replace("/", os.sep))
        pairs.append((full, cid, rel))
    return pairs


def resolve_resume_checkpoint(output_dir: str, resume_from_checkpoint: str | None) -> str | None:
    if not resume_from_checkpoint:
        return None
    out = Path(output_dir)
    if resume_from_checkpoint == "latest":
        ckpts = sorted(
            out.glob("checkpoint-*"),
            key=lambda p: int(p.name.rsplit("-", 1)[1]),
        )
        ckpts = [p for p in ckpts if p.is_dir() and (p / "progress.json").is_file()]
        return str(ckpts[-1]) if ckpts else None
    path = Path(resume_from_checkpoint)
    if not path.is_dir():
        path = out / path.name
    return str(path) if (path / "progress.json").is_file() else None


def load_class_index(class_index_path: str):
    with open(class_index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    num_classes = int(idx["num_classes"])
    id_to_style_key = idx["id_to_style_key"]
    style_key_to_id = idx["style_key_to_id"]
    id_to_init = idx.get("id_to_init_prompt", {})
    class_names = [
        id_to_init.get(str(i), id_to_style_key[str(i)]) for i in range(num_classes)
    ]
    return num_classes, class_names, style_key_to_id, id_to_style_key


def cache_latents_multiclass(
    accelerator,
    vae,
    pairs,
    num_classes: int,
    train_transforms,
    weight_dtype,
    train_batch_size: int,
):
    """Encode in metadata order; fail on missing images or invalid class IDs."""
    all_latents = []
    all_labels = []
    missing = []
    bad_class = []
    for full, cid, rel in pairs:
        if not os.path.isfile(full):
            missing.append(rel)
            continue
        if cid < 0 or cid >= num_classes:
            bad_class.append((rel, cid))
            continue
    if missing:
        raise RuntimeError(
            f"[strict] {len(missing)} images missing under train_data_dir. First: {missing[:3]}"
        )
    if bad_class:
        raise RuntimeError(
            f"[strict] invalid class_id for {len(bad_class)} rows. First: {bad_class[:3]}"
        )

    vae.eval()
    pixel_buf = []
    label_buf = []

    def flush_batch():
        nonlocal pixel_buf, label_buf, all_latents, all_labels
        if not pixel_buf:
            return
        stacked = torch.stack(pixel_buf, dim=0).to(accelerator.device, dtype=weight_dtype)
        with torch.no_grad():
            lat = vae.encode(stacked).latent_dist.mode() * vae.config.scaling_factor
        all_latents.append(lat.cpu())
        all_labels.extend(label_buf)
        pixel_buf = []
        label_buf = []

    for full, cid, _rel in tqdm(pairs, desc="Encoding Images (metadata order)"):
        img = Image.open(full).convert("RGB")
        pv = train_transforms(img)
        pixel_buf.append(pv)
        label_buf.append(cid)
        if len(pixel_buf) >= train_batch_size:
            flush_batch()
    flush_batch()

    all_latents = torch.cat(all_latents, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    if all_latents.shape[0] != len(pairs):
        raise RuntimeError("internal: latent count != metadata rows")
    return all_latents, all_labels


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=args.seed),
    )

    if accelerator.num_processes != 1:
        raise ValueError("This training recipe uses one process and one GPU.")
    if args.max_train_steps < 1 or args.checkpointing_steps < 1:
        raise ValueError("Training and checkpoint steps must be positive.")
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_json(Path(args.output_dir) / "effective_training_config.json", vars(args))

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model_id = args.pretrained_model_name_or_path
    local_only = os.path.isdir(model_id) and os.path.isfile(
        os.path.join(model_id, "model_index.json")
    )
    if local_only:
        logger.info(f"[LoRA] Local SD1.5 model: {model_id}")

    noise_scheduler = DDPMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        revision=args.revision,
        local_files_only=local_only,
    )

    logger.info(f"[LoRA] Loading base SD and LoRA from: {args.lora_path} (Scale: {args.lora_scale})")
    base_pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=weight_dtype,
        safety_checker=None,
        revision=args.revision,
        local_files_only=local_only,
    )
    base_pipe.load_lora_weights(args.lora_path, adapter_name="default")
    base_pipe.set_adapters(["default"], adapter_weights=[args.lora_scale])

    vae = base_pipe.vae.to(accelerator.device, dtype=weight_dtype)
    unet = base_pipe.unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder = base_pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    tokenizer = base_pipe.tokenizer

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.eval()

    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    multiclass = args.metadata_jsonl is not None
    class_names = None
    style_key_to_id = None
    id_to_style_key = None

    if multiclass:
        if not args.class_index_json:
            raise ValueError("Metadata training requires --class_index_json")
        num_classes, class_names, style_key_to_id, id_to_style_key = load_class_index(
            args.class_index_json
        )
        n_cls = args.n_cls if args.n_cls is not None else num_classes
        if n_cls != num_classes:
            raise ValueError(
                f"--n_cls ({n_cls}) must match num_classes ({num_classes}) in the class index"
            )
        pairs = load_metadata_pairs(args.metadata_jsonl, args.train_data_dir)
        logger.info(f"[multiclass] {len(pairs)} rows, n_cls={n_cls}")
        all_latents, all_labels = cache_latents_multiclass(
            accelerator,
            vae,
            pairs,
            num_classes,
            train_transforms,
            weight_dtype,
            train_batch_size=max(1, args.train_batch_size),
        )
        max_id = int(all_labels.max().item())
        if max_id + 1 > num_classes:
            raise RuntimeError(f"max class_id {max_id} >= num_classes {num_classes}")
    else:
        n_cls = 1
        dataset = load_dataset(
            "imagefolder",
            data_files={"train": os.path.join(args.train_data_dir, "**")},
        )

        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples["image"]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            return examples

        train_dataset = dataset["train"].with_transform(preprocess_train)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            return {
                "pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float()
            }

        temp_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            collate_fn=collate_fn,
        )

        logger.info("***** Caching Latents into RAM (imagefolder) *****")
        vae.eval()
        all_latents = []
        with torch.no_grad():
            for batch in tqdm(temp_dataloader, desc="Encoding Images"):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.mode() * vae.config.scaling_factor
                all_latents.append(latents.cpu())
        all_latents = torch.cat(all_latents, dim=0)
        all_labels = torch.zeros(all_latents.shape[0], dtype=torch.long)

    logger.info(f"Cached {all_latents.shape[0]} latents. Unloading VAE to free VRAM!")

    del vae
    del base_pipe
    gc.collect()
    torch.cuda.empty_cache()

    ram_dataset = TensorDataset(all_latents, all_labels)
    train_dataloader = DataLoader(
        ram_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    if len(train_dataloader) == 0:
        raise ValueError("Training data must contain at least one full batch.")

    if class_names:
        captions = list(class_names)
    else:
        caption = ", ".join(
            part.strip()
            for part in (args.customize_prefix, args.customize_suffix)
            if part and part.strip()
        )
        captions = [caption or "grayscale fingerprint spoof scan"]

    prompt_learner = PromptLearner.from_captions(
        tokenizer,
        text_encoder,
        captions,
        rank=args.residual_rank,
        residual_scale=args.residual_scale,
        kappa=args.residual_kappa,
        cosine_min=args.cosine_min,
        norm_ratio_min=args.norm_ratio_min,
        norm_ratio_max=args.norm_ratio_max,
        dtype=torch.float32,
    ).to(accelerator.device)

    del text_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(
        prompt_learner.parameters(), lr=args.learning_rate, weight_decay=1e-2, eps=1e-08
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    prompt_learner, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        prompt_learner, optimizer, train_dataloader, lr_scheduler
    )

    tracker_name = "prompt_learner_multiclass" if multiclass else "prompt_learner"
    if accelerator.is_main_process:
        accelerator.init_trackers(tracker_name, config=vars(args))

    def compute_snr(timesteps):
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[
            timesteps
        ].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)
        return (alpha / sigma) ** 2

    logger.info("***** Running training *****")
    global_step = 0
    first_epoch = 0
    resume_batch = 0
    resume_path = resolve_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if args.resume_from_checkpoint and not resume_path:
        raise FileNotFoundError("No complete training state was found for resume.")
    if resume_path:
        progress = json.loads((Path(resume_path) / "progress.json").read_text(encoding="utf-8"))
        accelerator.load_state(resume_path)
        global_step = progress["global_step"]
        first_epoch = progress["next_epoch"]
        resume_batch = progress["next_batch"]
        if global_step > args.max_train_steps:
            raise ValueError("Checkpoint step exceeds the requested training limit.")
        logger.info(f"Resumed full state at step {global_step}, epoch {first_epoch}, batch {resume_batch}")

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        initial=global_step,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        if global_step >= args.max_train_steps:
            break
        prompt_learner.train()
        train_loss = 0.0
        train_dataloader.set_epoch(epoch)
        offset = resume_batch if epoch == first_epoch else 0
        active_loader = accelerator.skip_first_batches(train_dataloader, offset) if offset else train_dataloader
        for step, batch in enumerate(active_loader, start=offset):
            with accelerator.accumulate(prompt_learner):
                latents = batch[0].to(accelerator.device, dtype=weight_dtype)
                cls_idx = batch[1].to(accelerator.device, dtype=torch.long)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with accelerator.autocast():
                    prompt_hidden_states, anc_loss, res_penalty = prompt_learner(cls_idx=cls_idx)
                    model_pred = unet(noisy_latents, timesteps, prompt_hidden_states).sample

                target = noise
                snr = compute_snr(timesteps)
                mse_loss_weights = (
                    torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    / snr
                )
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()
                loss = (
                    loss
                    + args.anchor_loss_weight * anc_loss
                    + args.residual_loss_weight * res_penalty
                )
                avg_loss = accelerator.gather(loss.repeat(bsz)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(prompt_learner.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "anchor_loss": anc_loss.detach(),
                        "residual_penalty": res_penalty.detach(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                    save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                    accelerator.save_state(str(save_path))
                    next_batch = step + 1
                    next_epoch = epoch
                    if next_batch >= len(train_dataloader):
                        next_epoch, next_batch = epoch + 1, 0
                    torch.save(accelerator.unwrap_model(prompt_learner).state_dict(), save_path / "final_prompt_learner.pt")
                    save_json(save_path / "progress.json", {"global_step": global_step,
                              "next_epoch": next_epoch, "next_batch": next_batch})

            progress_bar.set_postfix({"loss": loss.detach().item()})
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(prompt_learner)
        save_path = os.path.join(args.output_dir, "final_prompt_learner.pt")
        npz_save_path = os.path.join(args.output_dir, "final_prompt_learner.npz")

        torch.save(unwrapped.state_dict(), save_path)

        save_kw = dict(
            residual_a=unwrapped.residual_a.detach().cpu().float().numpy(),
            residual_b=unwrapped.residual_b.detach().cpu().float().numpy(),
            anchor=unwrapped.anchor.detach().cpu().float().numpy(),
            captions=np.array(unwrapped.captions, dtype=object),
            n_cls=np.array([unwrapped.n_cls], dtype=np.int64),
            residual_rank=np.array([unwrapped.rank], dtype=np.int64),
            residual_scale=unwrapped.residual_scale.detach().cpu().numpy(),
            residual_kappa=unwrapped.residual_kappa.detach().cpu().numpy(),
        )
        if multiclass and class_names is not None:
            save_kw["style_keys"] = np.array(class_names, dtype=object)

        np.savez(npz_save_path, **save_kw)

        if multiclass and style_key_to_id is not None:
            class_map_path = os.path.join(args.output_dir, "class_map.json")
            with open(class_map_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "style_key_to_id": style_key_to_id,
                        "id_to_style_key": id_to_style_key,
                        "num_classes": n_cls,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"Wrote {class_map_path}")

        logger.info(f"Training complete. Saved to {npz_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
