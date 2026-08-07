"""Load a trained checkpoint and generate text from it.

Kept separate from training so inference has no dependency on the training loop. The
generation loop crops the context to ``block_size`` each step — RoPE would extrapolate
past it, but to positions the model never saw during training.
"""
import torch
import torch.nn.functional as F

from slm import chat, checkpoint, paths
from slm.config import SEP, ModelConfig
from slm.model import JLM
from slm.tokenizer import Tokenizer, load_tokenizer, load_tokenizer_json

EOT = SEP                       # document boundary: the empty prompt, and the stop marker


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


def tokenizer_for(meta: dict, cfg: ModelConfig, tokenizer_path=None) -> Tokenizer:
    """The tokenizer a checkpoint was trained with, and the checks that it really is.

    Resolution order: an explicit ``tokenizer_path`` (the override, for a checkpoint that
    embeds none), then the copy embedded in the checkpoint, then the repo's current one.

    Lives here rather than in a CLI command because *every* consumer needs it. Written
    inline in ``generate`` it was inherited by nothing, and ``sft-eval`` scored the archived
    v1 model against the v2 tokenizer — same ``vocab_size``, so the one check it did run
    passed, and every id meant something else.
    """
    embedded = meta.get("tokenizer_json")
    if tokenizer_path is not None:
        tok = load_tokenizer(tokenizer_path)
    elif embedded:
        tok = load_tokenizer_json(embedded, source="tokenizer embedded in the checkpoint")
    else:
        tok = load_tokenizer(paths.TOKENIZER_PATH)

    assert cfg.vocab_size == tok.n_vocab, (
        f"checkpoint has vocab {cfg.vocab_size} but the tokenizer has {tok.n_vocab} — "
        f"pass --tokenizer for the one this model was trained with")
    recorded = meta.get("tokenizer_fingerprint")
    assert recorded is None or recorded == tok.fingerprint, (
        f"checkpoint was trained with tokenizer {recorded} but this one is "
        f"{tok.fingerprint} — every id would mean something else")
    if recorded is None:
        print("warning: checkpoint records no tokenizer; cannot verify the pairing")
    return tok


def is_chat_checkpoint(meta: dict) -> bool:
    """Whether the checkpoint's corpus was rendered as chat, i.e. whether to use ChatML.

    ``False`` when the checkpoint records no corpus at all — which is what every checkpoint
    written before :mod:`slm.train` started stamping one does, so an SFT model from before
    then still needs an explicit ``--chat``.
    """
    return (meta.get("corpus") or {}).get("render_kind") == "chat"


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filter: keep the smallest set of tokens whose cumulative probability
    reaches ``top_p``, zero the rest, and renormalize.

    Sampling from the full 32k-way softmax means the long tail of implausible tokens
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
           top_p: float = 0.9, block_size: int, device, stop_id: int | None = None):
    """Autoregressively sample ``max_new_tokens`` tokens continuing ``prompt``, yielding
    the continuation as it arrives. Stops early on ``stop_id``, which defaults to the
    document separator: the model has ended its document, and the corpus trains it to start
    an unrelated one straight after. A chat model stops on ``<|im_end|>`` instead.

    ``top_p >= 1.0`` disables nucleus filtering (sample from the full distribution).
    Temperature is applied first: it changes the probabilities, so it also changes which
    tokens fall inside the nucleus.
    """
    if stop_id is None:
        stop_id = tokenizer.sep_id(EOT)
    ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    pending = []

    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -block_size:])          # crop to the context window
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        if top_p < 1.0:
            probs = top_p_filter(probs, top_p)
        nxt = torch.multinomial(probs, num_samples=1)
        if int(nxt) == stop_id:
            return
        ids = torch.cat([ids, nxt], dim=1)
        pending.append(int(nxt))
        chunk = tokenizer.decode(pending)
        if not chunk.endswith("�"):                 # a character spanning two tokens
            yield chunk
            pending.clear()


def generate(model, tokenizer, **kwargs) -> str:
    """The whole continuation at once. See :func:`stream`."""
    return "".join(stream(model, tokenizer, **kwargs))


def fit_history(history: list[dict], *, tokenizer: Tokenizer, block_size: int,
                reserve: int, system: str | None = None) -> tuple[str, int, list[dict]]:
    """Render a conversation to a prompt, dropping oldest exchanges until the answer fits.

    Returns ``(prompt, n_prompt_tokens, kept)``. ``kept`` is the surviving history and the
    caller is expected to adopt it — trimming only a local copy leaves the real history
    growing without bound and re-encoded in full on every turn.

    Here rather than in the CLI because it is *policy*, not presentation: which turns the
    model gets to see is the same class of decision as :func:`slm.chat.fold_system`, and for
    the same reason — a second consumer that trimmed its own way would quietly show the model
    a context shape SFT never trained on. Not in :mod:`slm.chat`, which imports nothing and
    so cannot count tokens.

    The fold happens *after* trimming, so the system prompt lands on whichever user turn
    survives as the first rather than on one already dropped. A single turn too long to fit
    is returned anyway, over budget: the caller must warn, because :func:`stream` then crops
    the *head*, silently discarding the role header and the system prompt.
    """
    turns = list(history)
    while True:
        msgs = ([{"role": "system", "content": system}] if system else []) + turns
        prompt = chat.render_prompt(chat.fold_system(msgs), tokenizer.declared_specials)
        n = len(tokenizer.encode(prompt))
        if n + reserve <= block_size or len(turns) <= 1:
            return prompt, n, turns
        turns = turns[2:]               # oldest user+assistant pair
