# Jayden's Small Language Model

A 502M-parameter language model built from scratch — the transformer, the BPE tokenizer, the
training loop, the data pipeline. Pretrained on a subset of
[smolLM-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), then
instruction-tuned on [smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2).

The model code and tokenizer are hand-rolled; the default tokenizer at runtime is
HuggingFace's rust BPE (there is a from-scratch fallback), and attention uses PyTorch's
fused kernels.

## Run it

**1. Get the weights.** Download the latest checkpoint (~2 GB) from
[Releases](../../releases). The tokenizer is embedded in the file, so there is nothing else
to fetch and the weights cannot be paired with the wrong vocabulary.

**2. Set up.** Needs Python 3.10 and [uv](https://docs.astral.sh/uv/):

```bash
git clone <this repo> && cd slm
uv sync
```

**3. Talk to it.**

```bash
uv run main.py chat --checkpoint <checkpoint path>
```

`/reset` clears the conversation, `/exit` quits. Each turn prints its prompt-token count so
you can watch the 2048-token context fill; the oldest exchanges are dropped when it does.

A GPU is picked automatically — whichever has the most free memory. `--device cpu` works and
is slow but usable. About 1 GB of VRAM in bf16, 2 GB in fp32.

### Other things you can do

```bash
# one-shot answer, no conversation
uv run main.py generate --checkpoint jlm-502m-chat.pt --prompt "Explain photosynthesis"

# raw completion instead of chat (it continues your text rather than answering)
uv run main.py generate --checkpoint jlm-502m-chat.pt --prompt "The capital of France" --no-chat

# give it a persona
uv run main.py chat --checkpoint jlm-502m-chat.pt --system "You are a terse pirate."

# what's actually in the file: architecture, corpus, every training run behind it
uv run main.py inspect jlm-502m-chat.pt
```

Sampling defaults to `--temperature 1.0 --top-p 0.95`.

## What it is

| | |
|---|---|
| parameters | 502,191,616 |
| layers / hidden / heads | 16 / 1536 / 12 (head_dim 128) |
| context | 2048 tokens |
| vocabulary | 32,000 (BPE, 32 reserved special slots) |
| architecture | pre-norm decoder, RMSNorm, SwiGLU, RoPE, QK-norm, no biases, tied embeddings |
| chat format | ChatML (`<\|im_start\|>role … <\|im_end\|>`), no system role — a system prompt folds into the first user turn |
| precision | trained bf16-mixed |

## Results

| stage | corpus | tokens | val loss | val bpb |
|---|---|---|---|---|
| pretrain | smollm-corpus | 10.37B | 2.2515 | 0.698 |
| + SFT | smoltalk2 | 2.46B (2 epochs) | 1.0616 | — |

The two val numbers are **not comparable** — different corpora, and the SFT loss covers
assistant tokens only.

What the fine-tune bought, measured greedily on 40 held-out prompts:

| | base | after SFT |
|---|---|---|
| generations that loop | 87.5% | **2.5%** |
| generations that stop on their own | 0% | **35%** |
| ARC-Easy (200 items, chance 25%) | 47.0% | 46.0% |

So instruction tuning fixed the looping and taught it to stop, at essentially no cost in
knowledge.

## Example

```
you> Hello there!
bot> Hello! It's nice to meet you. What brings you here today? Are you looking for advice, help with a specific issue, or just someone to chat with for company?
[13 prompt tokens]

you> What is the biggest national park in california
bot> Ah, yes, there's the Big Sur National Park in California. It's a huge area that's both a natural wonder and a treasure trove of wildlife and adventure. It's known for its rugged coastline, diverse ecosystems, and majestic waterfalls. If you're in the right spirit, you can spend hours exploring the park's many trails, swimming in its many hot springs, and taking in its stunning natural beauty.

One of the main attractions is the Quinault Indian Village, which is situated on the mainland and offers a glimpse into the lives of Native American tribes that have lived here for centuries. It's a great place to learn about the region's rich cultural history and to enjoy a traditional Native American feast at the village.

Another highlight is the Anastasia Palisades Lodge, the largest public lodge in the state. It's an impressive venue for events like outdoor concerts, raffles, and art exhibits. And for those who love the thrill of hiking, the 263-acre Parkview Mountain State Park is a must-visit for its breathtaking views of the Bay Area.

Lastly, if you're looking for something a bit more off the beaten path, the Muir Woods National Monument in California is another incredible destination. It's a vast wilderness area with some of the most iconic and stunning natural features in the world, including giant sequoia groves, gorges, waterfalls, and volcanic landscapes.

All of these spots combine to make the Big Sur National Park a truly unforgettable experience. Would you like me to suggest a specific itinerary for you to explore on your next trip?
[69 prompt tokens]

you> write me a short story about a man who really likes cake
bot> In a quiet suburban home on the outskirts of the city, a 42-year-old man named Malcolm stood before his grand box of chocolate-coated Arbequins, carefully stacking them in a single layer. He had heard tales of these iconic French-style sugar cookies, but since he had never taken a bite, he was eager to experience their flavor for himself.

As he worked, his face illuminated with a warm, golden light, he was transported to a bygone era when French patisseries were the epitome of culinary sophistication. He loved the way the delicate, edible centers were adorned with intricate sugar flowers and gold leaf, and the rich, buttery flavor was like nothing he had ever tasted.

But beneath the exterior, a secret was brewing. Malcolm was secretly an artist, and he had always been fascinated by the way the delicate layers of sugar and chocolate blended together to create something truly magical. He began to experiment with his own creations, using techniques he had learned from his grandmother, who was an accomplished cake maker herself.

One day, while working on a particularly complex dessert, Malcolm's apartment suddenly felt like it was being stirred. He felt a chill run down his spine, and his mind, like an oven, was on high alert. His pastry chef, Marie, and her husband, Pierre, were trapped in a frantic race to get him off the sofa, but it was too late.

Malcolm's imagination took over, and he started to envision a perfect wedding cake, adorned with gold leaf and a vibrant flower. But just as he was about to pitch it to his lawyer to create the perfect marriage proposal, he heard a loud, sudden noise outside. He went to investigate and found a woman frantically standing on the steps of his apartment, her phone clutched in her hand.

With a mix of fear and hope, Malcolm sprang into action. He saw that the woman was a product of his art class, and in her case, it was a challenge. Without hesitation, he reached for the remaining ingredients and began to work the fragile cake on a few key design elements, using a napkin to serve as a makeshift plate.

As the final layer of the cake was added, Malcolm's hands trembled with anticipation, and for the first time in years, he felt a sense of purpose. He knew that this was his moment to create something truly beautiful, something that would capture the hearts of all who loved to eat. And
[426 prompt tokens]
```

## Training your own

Everything needed to reproduce this is in the repo. The corpora are not — they are built
locally and are ~56 GB.

```bash
uv run main.py prepare-data --corpus smollm       # encode the pretraining corpus (hours)
uv run main.py train --corpus smollm --out-dir checkpoints/run1 \
  --devices 8 --batch-size 12 --accumulate-grad-batches 2 --target-tokens 10e9 --wandb
uv run main.py prepare-data --corpus smoltalk2    # the SFT corpus
uv run main.py sft --init-from checkpoints/run1/step<N>-val<X>.ckpt \
  --out-dir checkpoints/sft --epochs 2
uv run main.py export checkpoints/sft/step<N>-val<X>.ckpt --out model.pt
```

`--help` on any command explains its flags.

