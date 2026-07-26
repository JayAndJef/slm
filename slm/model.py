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

from slm.config import ModelConfig

class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding."""

    def __init__(self, head_dim: int = 64):
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

    def _precompute_angles(self, head_dim: int = 64):
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

    def forward(self, x):
        B, T, D = x.shape
        head_dim = D // self.num_heads
        q = self.q_proj(x).reshape(B, T, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, head_dim).transpose(1, 2)
        q = self.rope(q)   # rotate queries and keys (not values) by position
        k = self.rope(k)
        out = self._attend_sdpa(q, k, v) if self.use_sdpa else self._attend_manual(q, k, v)
        out = out.transpose(1, 2).flatten(2, 3)
        return self.out_proj(out)

    def _attend_sdpa(self, q, k, v):
        """Fused causal attention: same math as :meth:`_attend_manual`, no (T, T) tensor."""
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def _attend_manual(self, q, k, v):
        """The longhand version: score -> causal mask -> softmax -> mix values."""
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / self.scale
        T = scores.size(-1)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # ty:ignore[not-subscriptable]
        scores = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", scores, v)
        return out
    

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, middle_size: int = 2048):
        super().__init__()
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
        self.ffn = SwiGLU(hidden_size=hidden_dim, middle_size=2048)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.attn(self.ln1(x)) + x
        x = self.ffn(self.ln2(x)) + x
        return x


class JLM(nn.Module):
    """Token embeddings -> N transformer blocks -> LM head. Positions come from RoPE
    inside attention, so there is no absolute position embedding.

    ``forward`` returns ``(logits, loss)``: logits are ``(B, T, vocab_size)``; loss is the
    next-token cross-entropy when ``targets`` is given, else ``None``.
    """

    def __init__(self, vocab_size: int = 4096, hidden_dim: int = 768,
                 num_heads: int = 12, n_layer: int = 12, block_size: int = 512,
                 use_sdpa: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(hidden_dim, num_heads, block_size, use_sdpa)
            for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "JLM":
        return cls(cfg.vocab_size, cfg.hidden_dim, cfg.num_heads, cfg.n_layer, cfg.block_size)

    def forward(self, x, targets=None):
        stream = self.embedding(x)   # positions come from RoPE inside attention
        logits = self.lm_head(self.norm(self.blocks(stream)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                torch.flatten(logits, 0, 1),
                torch.flatten(targets),
            )
        return logits, loss
