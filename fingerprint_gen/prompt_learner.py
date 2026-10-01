import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import logging

logger = logging.get_logger(__name__)

SEQ_LEN = 77
HIDDEN_DIM = 768
RESIDUAL_RANK = 8


def encode_captions(tokenizer, text_encoder, captions):
    """Encode each caption once with frozen CLIP; return anchors shaped (n_cls, 77, 768)."""
    text_encoder.eval()
    device = next(text_encoder.parameters()).device
    tokens = tokenizer(
        list(captions),
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    with torch.no_grad():
        hidden = text_encoder(input_ids)[0]
    return hidden.detach()


class PromptLearner(nn.Module):
    """Learn a bounded low-rank residual around a cached CLIP caption anchor.

    D = A B, with A shaped (n_cls, 77, r), B shaped (n_cls, r, 768), and r = 8.
    Clip the residual relative to the anchor norm: H = E0 + s * clip(D).
    Each combination has r * (77 + 768) = 6760 trainable parameters.
    """

    def __init__(
        self,
        anchor,
        rank=RESIDUAL_RANK,
        residual_scale=1.0,
        kappa=0.25,
        cosine_min=0.97,
        norm_ratio_min=0.9,
        norm_ratio_max=1.1,
        dtype=torch.float32,
        captions=None,
    ):
        super().__init__()
        if anchor.dim() != 3:
            raise ValueError(f"Expected anchor shape (n_cls, seq, dim), received {tuple(anchor.shape)}")

        n_cls, seq_len, hidden_dim = anchor.shape
        self.n_cls = int(n_cls)
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.cosine_min = float(cosine_min)
        self.norm_ratio_min = float(norm_ratio_min)
        self.norm_ratio_max = float(norm_ratio_max)
        self.captions = list(captions) if captions is not None else [""] * self.n_cls
        if len(self.captions) != self.n_cls:
            raise ValueError(
                f"Caption count {len(self.captions)} differs from n_cls {self.n_cls}"
            )

        self.register_buffer("anchor", anchor.detach().to(dtype=dtype))
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale), dtype=torch.float32))
        self.register_buffer("residual_kappa", torch.tensor(float(kappa), dtype=torch.float32))

        factor_a = torch.empty(self.n_cls, self.seq_len, self.rank, dtype=dtype)
        nn.init.normal_(factor_a, std=0.02)
        factor_b = torch.zeros(self.n_cls, self.rank, self.hidden_dim, dtype=dtype)
        self.residual_a = nn.Parameter(factor_a)
        self.residual_b = nn.Parameter(factor_b)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "PromptLearner: n_cls=%s, rank=%s, seq=%s, dim=%s, trainable=%s",
            self.n_cls,
            self.rank,
            self.seq_len,
            self.hidden_dim,
            n_trainable,
        )

    @classmethod
    def from_captions(
        cls,
        tokenizer,
        text_encoder,
        captions,
        rank=RESIDUAL_RANK,
        residual_scale=1.0,
        kappa=0.25,
        cosine_min=0.97,
        norm_ratio_min=0.9,
        norm_ratio_max=1.1,
        dtype=torch.float32,
    ):
        if not captions:
            raise ValueError("At least one caption is required")
        anchor = encode_captions(tokenizer, text_encoder, captions)
        return cls(
            anchor=anchor,
            rank=rank,
            residual_scale=residual_scale,
            kappa=kappa,
            cosine_min=cosine_min,
            norm_ratio_min=norm_ratio_min,
            norm_ratio_max=norm_ratio_max,
            dtype=dtype,
            captions=list(captions),
        )

    @classmethod
    def from_checkpoint(cls, path, device="cpu", dtype=None):
        state = torch.load(path, map_location="cpu")
        if "anchor" not in state or "residual_a" not in state:
            raise KeyError(f"Checkpoint is missing anchor / residual_a: {path}")
        module = cls(
            anchor=state["anchor"],
            rank=int(state["residual_a"].shape[-1]),
            residual_scale=float(state["residual_scale"]),
            kappa=float(state["residual_kappa"]),
            dtype=state["anchor"].dtype,
        )
        module.load_state_dict(state, strict=True)
        if dtype is not None:
            module = module.to(dtype=dtype)
        return module.to(device)

    def _index(self, cls_idx):
        if cls_idx is None:
            cls_idx = torch.zeros(1, dtype=torch.long, device=self.anchor.device)
        elif not isinstance(cls_idx, torch.Tensor):
            cls_idx = torch.tensor(cls_idx, dtype=torch.long, device=self.anchor.device)
        else:
            cls_idx = cls_idx.to(device=self.anchor.device, dtype=torch.long)
        if cls_idx.ndim == 0:
            cls_idx = cls_idx.view(1)
        return cls_idx

    def residual(self, cls_idx):
        """Unclipped low-rank residual D = A B, shaped (batch, 77, 768)."""
        cls_idx = self._index(cls_idx)
        return torch.matmul(self.residual_a[cls_idx], self.residual_b[cls_idx])

    def clip_residual(self, delta, anchor):
        """||D̃||_F ≤ κ ||E0||_F。"""
        flat_delta = delta.float().reshape(delta.shape[0], -1)
        flat_anchor = anchor.float().reshape(anchor.shape[0], -1)
        delta_norm = flat_delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
        anchor_norm = flat_anchor.norm(dim=1, keepdim=True)
        limit = self.residual_kappa.float() * anchor_norm
        scale = torch.clamp(limit / delta_norm, max=1.0)
        clipped = (flat_delta * scale).reshape_as(delta)
        return clipped.to(dtype=delta.dtype)

    def hidden_state(self, cls_idx):
        """H = E0 + s * clipped D. Return (hidden states, unclipped residual)."""
        cls_idx = self._index(cls_idx)
        anchor = self.anchor[cls_idx]
        delta = torch.matmul(self.residual_a[cls_idx], self.residual_b[cls_idx])
        clipped = self.clip_residual(delta, anchor)
        scale = self.residual_scale.to(device=anchor.device, dtype=anchor.dtype)
        hidden = anchor + scale * clipped
        return hidden, delta

    def anchor_regularizer(self, hidden, anchor):
        """Penalize cosine below its lower bound and norm ratio outside its interval."""
        flat_h = hidden.float().reshape(hidden.shape[0], -1)
        flat_e = anchor.float().reshape(anchor.shape[0], -1)
        cosine = F.cosine_similarity(flat_h, flat_e, dim=1)
        cosine_pen = F.relu(self.cosine_min - cosine)
        ratio = flat_h.norm(dim=1) / flat_e.norm(dim=1).clamp_min(1e-12)
        ratio_pen = F.relu(ratio - self.norm_ratio_max) + F.relu(self.norm_ratio_min - ratio)
        return (cosine_pen.square() + ratio_pen.square()).mean()

    def forward(self, cls_idx=None):
        cls_idx = self._index(cls_idx)
        hidden, delta = self.hidden_state(cls_idx)
        anc_loss = self.anchor_regularizer(hidden, self.anchor[cls_idx])
        res_penalty = delta.float().square().sum(dim=(1, 2)).mean()
        return hidden, anc_loss, res_penalty

    @torch.no_grad()
    def condition(self, cls_idx=0, device=None, dtype=None):
        """Deterministic inference condition, shaped (1 or batch, 77, 768)."""
        hidden, _delta = self.hidden_state(cls_idx)
        if device is not None or dtype is not None:
            hidden = hidden.to(
                device=device if device is not None else hidden.device,
                dtype=dtype if dtype is not None else hidden.dtype,
            )
        return hidden
