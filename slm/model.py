"""The JLM transformer: a GPT-style decoder-only language model, from scratch.

Composition: :class:`MultiheadSelfAttention` -> :class:`TransformerBlock` -> :class:`JLM`.
The module names and buffer layout here are load-bearing — a trained checkpoint stores a
``state_dict`` keyed by them, so renaming a submodule would break ``load_state_dict``.

This module depends only on :mod:`torch` and :class:`ModelConfig`; it knows nothing
about data, training, or the filesystem. Position information is injected by rotary
embeddings (RoPE) inside attention, so there is no absolute position embedding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from slm.config import ModelConfig

_compiled_create_block_mask = torch.compile(create_block_mask, dynamic=False)


def segment_block_mask(seg: torch.Tensor, compiled: bool = True) -> BlockMask:
    """Causal mask further restricted to attention *within* a segment.

    ``seg`` is ``(B, T)``: a per-token segment index. This module stays deliberately
    agnostic about what a segment means — documents in pretraining, prompt/answer pairs
    under supervised fine-tuning — so switching datasets never touches the model.

    Invariant: the diagonal is always allowed (``q >= q`` and ``seg[q] == seg[q]``), so no
    query row can be fully masked. A future ``mask_mod`` that breaks this would produce
    NaNs, not an error.
    """
    B, T = seg.shape

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (seg[b, q_idx] == seg[b, kv_idx])

    build = _compiled_create_block_mask if compiled else create_block_mask
    return build(mask_mod, B, None, T, T, device=seg.device)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding."""

    def __init__(self, head_dim: int):
        super().__init__()
        # Derived constant (not learned); persistent=False keeps it out of the state_dict.
        self.register_buffer("freqs", self._precompute_angles(head_dim), persistent=False)

    def forward(self, x: torch.Tensor):
        B, H, T, D = x.shape
        rotation_table = torch.einsum("i,j->ij", torch.arange(T, device=x.device), self.freqs)
        cos = torch.cos(rotation_table).to(x.dtype)   # match x's dtype (bf16 under autocast)
        sin = torch.sin(rotation_table).to(x.dtype)

        transformed_x = x.reshape(B, H, T, D // 2, 2)
        x_a = transformed_x[..., 0]
        x_b = transformed_x[..., 1]
        x_a_rot = x_a * cos - x_b * sin
        x_b_rot = x_a * sin + x_b * cos
        return torch.stack([x_a_rot, x_b_rot], dim=-1).reshape(B, H, T, D)

    def _precompute_angles(self, head_dim: int):
        freqs = 1 / (10000 ** (torch.arange(0, head_dim, 2)/head_dim))
        return freqs
        


class MultiheadSelfAttention(nn.Module):
    """Causal multi-head self-attention.

    Projects the input to per-head queries/keys/values, computes scaled dot-product
    scores, masks out future positions (lower-triangular ``tril`` buffer), softmaxes,
    and mixes the values. A final projection lets the heads exchange information.

    Two interchangeable implementations of that middle step, selected by ``use_sdpa``:
    :meth:`_attend_manual` is the from-scratch version written out longhand, and
    :meth:`_attend_sdpa` defers to PyTorch's fused kernel. They compute the same
    function — ``is_causal=True`` is the same lower-triangular mask (q and k are always
    the same length here) and SDPA's default ``scale`` is ``1/sqrt(head_dim)``, matching
    ``self.scale`` — but the fused kernel never materializes the ``(B, H, T, T)`` score
    matrix, so it uses far less memory.
    """

    def __init__(self, hidden_dim: int, num_heads: int, block_size: int,
                 use_sdpa: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.scale = (hidden_dim // num_heads) ** 0.5
        self.rope = RotaryPositionalEmbedding(hidden_dim // num_heads)
        self.use_sdpa = use_sdpa

    def forward(self, x, block_mask=None):
        """``block_mask`` (a FlexAttention ``BlockMask``) restricts attention beyond plain
        causality — see :func:`segment_block_mask`. When one is given it wins over
        ``use_sdpa``: both other paths are causal-only, so do not "verify" the flex path
        against :meth:`_attend_manual` — that compares two different functions.
        """
        B, T, D = x.shape
        head_dim = D // self.num_heads
        q = self.q_proj(x).reshape(B, T, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, head_dim).transpose(1, 2)
        q = self.rope(q)   # rotate queries and keys (not values) by position
        k = self.rope(k)
        if block_mask is not None:
            out = self._attend_flex(q, k, v, block_mask)
        else:
            out = self._attend_sdpa(q, k, v) if self.use_sdpa else self._attend_manual(q, k, v)
        out = out.transpose(1, 2).flatten(2, 3)
        return self.out_proj(out)

    def _attend_flex(self, q, k, v, block_mask):
        """Fused attention under an arbitrary block mask. Default scale is
        ``1/sqrt(head_dim)``, matching :meth:`_attend_sdpa` and ``self.scale``."""
        return flex_attention(q, k, v, block_mask=block_mask)

    def _attend_sdpa(self, q, k, v):
        """Fused causal attention: same math as :meth:`_attend_manual`, no (T, T) tensor."""
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def _attend_manual(self, q, k, v):
        """The longhand version: score -> causal mask -> softmax -> mix values.
        """
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / self.scale
        T = scores.size(-1)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # ty:ignore[not-subscriptable]
        scores = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", scores, v)
        return out
    

class SwiGLU(nn.Module):
    """Gated FFN: ``w3(silu(w1 x) * w2 x)``.

    ``middle_size`` defaults to ``8/3 * hidden_size`` rounded up to a multiple of 128 —
    three matrices at 8/3 width hold the same parameter count as the two-matrix 4x GELU
    MLP it replaces (``3 * d * 8d/3 == 8d^2``), so swapping FFNs is not secretly also a
    model-size change. Derived rather than hardcoded: a fixed value silently changes the
    ratio the moment ``hidden_dim`` moves.
    """

    def __init__(self, hidden_size: int, middle_size: int | None = None):
        super().__init__()
        if middle_size is None:
            middle_size = -(-(8 * hidden_size // 3) // 128) * 128   # ceil to a 128 multiple
        self.w1 = nn.Linear(hidden_size, middle_size)
        self.w2 = nn.Linear(hidden_size, middle_size)
        self.w3 = nn.Linear(middle_size, hidden_size)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x + attn(ln1(x)), then x + ffn(ln2(x))."""

    def __init__(self, hidden_dim: int, num_heads: int, block_size: int,
                 use_sdpa: bool = True):
        super().__init__()
        self.attn = MultiheadSelfAttention(hidden_dim, num_heads, block_size, use_sdpa)
        self.ffn = SwiGLU(hidden_dim)   # width derived from hidden_dim, not pinned
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, block_mask=None):
        x = self.attn(self.ln1(x), block_mask) + x
        x = self.ffn(self.ln2(x)) + x
        return x


class JLM(nn.Module):
    """Token embeddings -> N transformer blocks -> LM head. Positions come from RoPE
    inside attention, so there is no absolute position embedding.

    ``forward`` returns ``(logits, loss)``: logits are ``(B, T, vocab_size)``; loss is the
    next-token cross-entropy when ``targets`` is given, else ``None``.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_heads: int, n_layer: int,
                 block_size: int, use_sdpa: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # ModuleList, not Sequential: blocks take a second (mask) argument, which
        # Sequential cannot forward. State_dict keys ``blocks.N.*`` are unchanged.
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, block_size, use_sdpa)
            for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "JLM":
        return cls(cfg.vocab_size, cfg.hidden_dim, cfg.num_heads, cfg.n_layer, cfg.block_size)

    def forward(self, x, targets=None, block_mask=None):
        """``block_mask`` defaults to ``None`` -> plain causal attention, which is what
        generation uses (no segment information exists there).

        Targets equal to ``-100`` are excluded from the loss. Nothing produces them during
        pretraining (token ids are unsigned); it is the hook for SFT, where the loss should
        cover answer tokens only.
        """
        stream = self.embedding(x)   # positions come from RoPE inside attention
        for blk in self.blocks:
            stream = blk(stream, block_mask)
        logits = self.lm_head(self.norm(stream))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                torch.flatten(logits, 0, 1),
                torch.flatten(targets),
                ignore_index=-100,
            )
        return logits, loss
