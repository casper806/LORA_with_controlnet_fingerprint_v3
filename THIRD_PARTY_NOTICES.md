# Third-party components

`diffusers/examples/text_to_image/train_text_to_image_lora.py` is adapted
from the Hugging Face diffusers text-to-image LoRA training example.
Its original copyright and Apache License 2.0 notice are retained.
The adaptations provide an explicit training cursor for checkpoint resume,
seeded data sampling, a hard stopping limit, and pinned model revisions.

- Upstream project: https://github.com/huggingface/diffusers
- Upstream example: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
- License text: [Apache License 2.0](licenses/Apache-2.0.txt)

Dependencies installed from `requirements.txt` retain their own licenses.
Stable Diffusion and ControlNet model weights are obtained separately and
retain the terms published with their respective model repositories.

A license for the project-authored code has not been specified.
