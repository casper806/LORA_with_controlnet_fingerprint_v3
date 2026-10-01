# Local model snapshots

Place complete diffusers-format snapshots here, or set `LOCAL_MODEL_DIR`
and `LOCAL_CONTROLNET_DIR` to external absolute paths.

```text
hf_models/
  stable-diffusion-v1-5/
    model_index.json
    scheduler/
    tokenizer/
    text_encoder/
    unet/
    vae/
  sd-controlnet-canny/
    config.json
    diffusion_pytorch_model.safetensors
```

The configured Hub model identifiers are in
[model_profile.json](../configs/livdet2015/model_profile.json).
Local snapshots are identified by file hashes in run configurations.
An explicitly set local directory must be valid; it is not silently replaced
by another model. Downloaded snapshots and trained weights are excluded from Git.

