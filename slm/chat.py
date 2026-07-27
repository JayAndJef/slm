"""The special-token block and the ChatML template.

Two responsibilities, both pure text:

- :func:`specials` builds the reserved token block a tokenizer declares at training time,
  and :func:`is_reserved` reports which of those slots are still unused.
- :func:`render_training` turns a conversation into ``(text, is_target)`` spans, and
  :func:`render_prompt` joins the non-answer ones into the string generation feeds the model.

This module imports **nothing** — not torch, not numpy, not ``datasets``, not the rest of
``slm``. That is what lets both the corpus builder and :mod:`slm.generate` depend on it: the
template must be one artifact, because a training/inference mismatch is invisible. The model
simply receives a prompt shape it never saw and produces slightly worse text forever.
"""

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


def render_training(messages: list[dict]) -> list[tuple[str, bool]]:
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
        spans.append((f"{IM_START}{m['role']}\n", False))
        spans.append((m["content"], m["role"] == "assistant"))
        spans.append((f"{IM_END}\n", m["role"] == "assistant"))
    return spans


def render_prompt(messages: list[dict]) -> str:
    """The exact string generation feeds the model, ending after the assistant header.

    Derived from :func:`render_training` rather than written as a second format string —
    that is the whole point of the two living in one module. Two independent templates drift,
    and nothing reports it.
    """
    spans = render_training(list(messages) + [{"role": "assistant", "content": ""}])
    return "".join(text for text, _ in spans[:-2])      # drop the empty answer and its IM_END
