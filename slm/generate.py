"""Load a trained checkpoint and generate text from it.

Kept separate from training so inference has no dependency on the training loop. The
generation loop crops the context to ``block_size`` each step — RoPE would extrapolate
past it, but to positions the model never saw during training.
"""
import torch
import torch.nn.functional as F

from slm import checkpoint
from slm.config import ModelConfig
from slm.model import JLM

EOT = "\n<|endoftext|>\n"       # document boundary: the empty prompt, and the stop marker


def load_model(checkpoint_path, device) -> tuple[JLM, ModelConfig, dict]:
    """Rebuild a :class:`JLM` from a checkpoint and load its weights (eval mode).

    Returns ``(model, model_cfg, meta)`` where ``meta`` carries ``step``/``val`` for
    display. The on-disk format lives in :mod:`slm.checkpoint`.
    """
    state, cfg, meta = checkpoint.load(checkpoint_path, map_location=device)
    model = JLM.from_config(cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg, meta


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filter: keep the smallest set of tokens whose cumulative probability
    reaches ``top_p``, zero the rest, and renormalize.

    Sampling from the full 4096-way softmax means the long tail of implausible tokens
    still gets picked occasionally — and one bad token stays in the context and degrades
    everything after it. Unlike top-k the kept set is adaptive: narrow where the model is
    confident, wide where it genuinely isn't.
    """
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    prior = sorted_probs.cumsum(dim=-1) - sorted_probs
    sorted_probs = sorted_probs * (prior < top_p)
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)


@torch.no_grad()
def stream(model, tokenizer, *, prompt: str = EOT,
           max_new_tokens: int = 250, temperature: float = 0.8,
           top_p: float = 0.9, block_size: int, device):
    """Autoregressively sample ``max_new_tokens`` tokens continuing ``prompt``, yielding
    the continuation as it arrives. Stops early when the model emits :data:`EOT`: it has
    ended its document, and the corpus trains it to start an unrelated one straight after.

    ``top_p >= 1.0`` disables nucleus filtering (sample from the full distribution).
    Temperature is applied first: it changes the probabilities, so it also changes which
    tokens fall inside the nucleus.
    """
    stop_id = tokenizer.sep_id(EOT)
    ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    new, emitted = [], 0

    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -block_size:])          # crop to the context window
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        if top_p < 1.0:
            probs = top_p_filter(probs, top_p)
        nxt = torch.multinomial(probs, num_samples=1)
        if int(nxt) == stop_id:
            return
        ids = torch.cat([ids, nxt], dim=1)
        new.append(int(nxt))
        text = tokenizer.decode(new)
        if not text.endswith("�"):                  # a character spanning two tokens
            yield text[emitted:]
            emitted = len(text)


def generate(model, tokenizer, **kwargs) -> str:
    """The whole continuation at once. See :func:`stream`."""
    return "".join(stream(model, tokenizer, **kwargs))
