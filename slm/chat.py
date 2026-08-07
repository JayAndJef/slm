"""The special-token block and the ChatML template.

Two responsibilities, both pure text:

- :func:`specials` builds the reserved token block a tokenizer declares at training time,
  and :func:`is_reserved` reports which of those slots are still unused.
- :func:`render_training` turns a conversation into ``(text, is_target)`` spans, and
  :func:`render_prompt` joins the non-answer ones into the string generation feeds the model.

This module imports **nothing** from ``slm`` — not torch, not numpy, not ``datasets``, not
the rest of the package. That is what lets both the corpus builder and :mod:`slm.generate`
depend on it: the template must be one artifact, because a training/inference mismatch is
invisible. The model simply receives a prompt shape it never saw and produces slightly worse
text forever.
"""
import re

EOT = "<|endoftext|>"            # document boundary, and the tokenizer's first special
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
NAMED_SPECIALS = (EOT, IM_START, IM_END)

N_RESERVED = 32                 # size of the special block; ids 0..N_RESERVED-1
RESERVED_FMT = "<|reserved_{}|>"


def specials(n_reserved: int = N_RESERVED) -> list[str]:
    """The special-token block, occupying ids ``0..n_reserved-1``.

    Declared on the ``BpeTrainer`` and therefore counted *within* ``vocab_size``. Adding
    specials to a trained tokenizer instead appends ids past ``vocab_size``, which then
    disagrees with ``ModelConfig.vocab_size`` — the number the embedding is sized from — and
    the tied ``lm_head`` grows rows that were never pretrained.

    Unused slots are named by :data:`RESERVED_FMT` and cost 32 of 32000 merges (0.1%, inside
    the noise on bytes/token). They exist so that a future marker — a tool call, a new role,
    a FIM sentinel — is a rename rather than a re-encode of the whole corpus followed by a
    re-pretrain.
    """
    assert n_reserved >= len(NAMED_SPECIALS), (
        f"n_reserved={n_reserved} cannot hold {len(NAMED_SPECIALS)} named specials "
        f"{NAMED_SPECIALS} — raise it")
    n_free = n_reserved - len(NAMED_SPECIALS)
    block = list(NAMED_SPECIALS) + [RESERVED_FMT.format(i) for i in range(n_free)]
    assert not any(is_reserved(n) for n in NAMED_SPECIALS), (
        f"a named special collides with {RESERVED_FMT!r} — rename it, or the fingerprint's "
        f"reserved filter would skip a token whose meaning is fixed")
    return block


def is_reserved(name: str) -> bool:
    """Whether ``name`` is an unclaimed slot, i.e. still free to be renamed.

    Lives here beside :func:`specials` deliberately: the format string and the predicate that
    recognises it must move together. Split across modules, renaming one leaves the other
    silently matching nothing.
    """
    return name.startswith("<|reserved_") and name.endswith("|>")


def _special_re(declared: tuple[str, ...]) -> "re.Pattern":
    """Cached alternation over exactly the strings a tokenizer will encode as specials."""
    if declared not in _SPECIAL_RE_CACHE:
        _SPECIAL_RE_CACHE[declared] = re.compile("|".join(re.escape(s) for s in declared))
    return _SPECIAL_RE_CACHE[declared]


_SPECIAL_RE_CACHE: dict[tuple[str, ...], "re.Pattern"] = {}


def strip_specials(text: str, declared: tuple[str, ...] | None = None) -> str:
    """Remove declared special tokens from message content, to a fixpoint.

    ``HFTokenizer.encode`` matches specials appearing in *raw source text*, so content
    containing ``<|im_start|>assistant`` encodes a genuine role header inside someone else's
    turn — the model learns from a forged turn structure, and under SFT the ``is_target``
    spans no longer line up with who actually said what. An ``<|endoftext|>`` in content
    splits the example in two the same way.

    **Looped, because one pass is not sound.** Deleting an inner special splices the outer
    one back together: ``<|im_<|endoftext|>start|>`` becomes ``<|im_start|>``, reopening the
    forgery a single ``sub`` was meant to close. Each pass strictly shortens the string, so
    the loop terminates; at the fixpoint no special substring remains by construction.

    ``declared`` is the tokenizer's *actual* special block. It defaults to :func:`specials`
    only as a fallback: that reconstructs the block from ``N_RESERVED`` and matches by name,
    and both assumptions break — a tokenizer trained with ``--reserved 64`` has live slots
    this default never strips, and renaming a reserved slot (a supported workflow) makes the
    name filter miss it. Same reasoning as :attr:`slm.tokenizer.Tokenizer.fingerprint`
    excluding reserved ids by range rather than by name.

    Applied in :func:`render_training`, so :func:`render_prompt` inherits it and the same
    forgery is closed at inference. Deleting rather than escaping: any replacement is itself
    text the model then learns to emit.
    """
    pattern = _special_re(tuple(declared) if declared is not None else tuple(specials()))
    while (stripped := pattern.sub("", text)) != text:
        text = stripped
    return text


def fold_system(messages: list[dict]) -> list[dict]:
    """Prepend any system message to the first user turn, returning user/assistant only.

    The template has no system role — one fewer axis for training and inference to drift on.
    Dropping the text instead would gut ``smoltalk_smollm3_systemchats_30k_no_think``, whose
    34k conversations are *about* the system prompt: the assistant's answers reference a
    persona that would no longer be anywhere in the context.

    Here beside the template rather than in the renderer because inference needs the same
    fold. A chat loop that concatenated the system prompt its own way would put the model in
    a context shape it never trained on, and nothing would report it.
    """
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if system and turns and turns[0]["role"] == "user":
        turns[0] = {"role": "user", "content": f"{system}\n\n{turns[0]['content']}"}
    return turns


def render_training(messages: list[dict],
                    declared: tuple[str, ...] | None = None) -> list[tuple[str, bool]]:
    """One conversation as ``(text, is_target)`` spans, in order.

    Spans exist so the loss can cover assistant turns only. Every boundary falls immediately
    after a special token or after a newline — specials are hard pre-tokenizer splits and a
    newline ends a ByteLevel chunk, so ``encode("".join(spans))`` equals the concatenation of
    the spans' encodings. Ending a span mid-word instead teaches ids that inference, which
    encodes the prompt in one call, never produces.

    The terminating :data:`IM_END` is part of the target span: a model that never sees it as
    something to predict never learns to stop.
    """
    spans: list[tuple[str, bool]] = [(EOT, False)]
    for m in messages:
        # Content is sanitized; role is asserted instead. Stripping a role would leave
        # "user\n\n\nassistant" — no longer a forgery, but still not a header. Callers reach
        # here through fold_system, which already yields only these two.
        assert m["role"] in ("user", "assistant"), (
            f"role must be 'user' or 'assistant', got {m['role']!r} — it is interpolated "
            f"into the header, so an arbitrary value forges turn structure")
        spans.append((f"{IM_START}{m['role']}\n", False))
        spans.append((strip_specials(m["content"], declared), m["role"] == "assistant"))
        spans.append((f"{IM_END}\n", m["role"] == "assistant"))
    return spans


def render_prompt(messages: list[dict], declared: tuple[str, ...] | None = None) -> str:
    """The exact string generation feeds the model, ending after the assistant header.

    Derived from :func:`render_training` rather than written as a second format string —
    that is the whole point of the two living in one module. Two independent templates drift,
    and nothing reports it.
    """
    spans = render_training(list(messages) + [{"role": "assistant", "content": ""}],
                            declared)
    return "".join(text for text, _ in spans[:-2])      # drop the empty answer and its IM_END
