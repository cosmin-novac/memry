# What Memry costs and what it saves

A measured answer to "why should I run a memory layer at all". The short version:
Memry spends cheap-model tokens once, at ingestion, to make every later recall
cheaper on the expensive model and, more importantly, correct across sessions.
The break-even is real and it is not at message one.

Numbers below come from `evals/cost_measure.py`: the real `MemoryStore`
pipeline driven by a scripted 4-session, 14-message conversation (duplicates,
one contradiction, recurring entities), with a fake LLM that answers every
prompt type plausibly and counts calls and tokens (`len // 4`, the same
estimator Memry uses). Absolute token counts are estimates; the ratios and the
call counts are exact. Re-run it after changing the pipeline.

## 1. Ingestion: what a saved message costs

Per user message, `store.add(infer=True)` made on average:

| Phase | LLM calls / msg | Input tokens / msg | Share of input |
|---|---:|---:|---:|
| Extraction (facts, type, importance, entities) | 1.0 | 1,220 | 51% |
| Entity identity checks | 1.7 | 630 | 26% |
| Reconciliation (per candidate fact) | 0.9 | 380 | 16% |
| Coverage audit | 0.8 | 160 | 7% |
| **Total** | **4.4** | **~2,400 in, ~125 out** | |

Two facts about that 2,400 matter more than the number itself:

- **85% of it is fixed system prompt** (28.5k of 33.5k tokens over the run).
  The variable part, your actual text plus retrieved neighbours, is ~360
  tokens per message. Whether the fixed part is cheap depends on prompt
  caching, and here Memry is currently unlucky: its system prompts are
  ~250-1,200 tokens each, below the cacheable minimum on Claude Haiku 4.5
  (4,096 tokens) and mostly below Sonnet 5's (1,024). Only the extraction
  prompt just clears Sonnet's bar. So on the models people actually pick for
  ingestion, assume the full price. (OpenAI's automatic caching starts at 1,024
  tokens, same story.)
- **The MCP path is half the price.** `save_memories` uses `add_deferred`; the
  background worker groups a session's saves into one extraction call and skips
  the coverage audit: 2.7 calls and ~1,300 input tokens per message in the same
  scenario. Growing the history to 104 messages held at ~2,100 input tokens and
  3.2 calls per message for the synchronous path.

Embeddings: ~2 texts per message (the message plus each stored fact), batched;
free with the default hash embedder, one HTTP call per batch with OpenAI.

In money, per saved message (per-message path / MCP path):

| Ingestion model | Price (in / out per MTok) | Cost per message |
|---|---|---:|
| Claude Haiku 4.5 | $1 / $5 | $0.0030 / $0.0019 |
| gpt-5-mini | ~$0.25 / $2 (check current list) | $0.0009 / $0.0006 |

A user saving 100 messages a day pays roughly $0.06-0.30/day. Small, but not
zero, and it is paid whether or not the memory is ever read.

Wall clock: the calls are sequential, so a real provider costs 3-25 s per
message. On the MCP path that runs in the background worker and the agent's
turn does not wait; on the Python `add()` path it blocks.

## 2. Read side: what a recall costs and what it replaces

`reconstruct_context(query, token_budget=1200)` returned 327-351 tokens
(13 memories) against the 14-message store, in 4-20 ms locally with hash
embeddings (add ~100-300 ms for a remote embedding call). Of those ~340
tokens, ~130 are formatting (header, footer, `- [type · date]` prefix per
line): the facts themselves are 209 tokens. That overhead is a lever, see §5.

What that replaces depends on what you would otherwise do:

| Alternative | Tokens injected per recall | Notes |
|---|---:|---|
| Nothing (no cross-session memory) | 0 | The information is simply gone next session. The cost is re-explaining, or wrong assumptions. |
| Paste the whole history | grows linearly | 343 tokens at 14 messages, 3,832 at 104 in this scenario; real chats with assistant turns are several times larger |
| Verbatim RAG over raw messages (Memry with no LLM) | 472 | Cheap to build, but for "where does Ada live" the top hit was the stale Munich message; Memry's distilled store returned Amsterdam because the move superseded it |
| Memry distilled context | 340 → ~720 (cap) | Saturates at 20 memories; stayed at ~720 tokens from 34 to 104 messages |

The scaling run makes the shape clear:

| Messages in history | Paste-history tokens | Memry context tokens | Recall latency (median) |
|---:|---:|---:|---:|
| 14 | 343 | 341 | 9 ms |
| 34 | 1,115 | 705 | 12 ms |
| 54 | 1,893 | 723 | 16 ms |
| 74 | 2,670 | 721 | 15 ms |
| 104 | 3,832 | 722 | 37 ms |

At 14 messages Memry saves nothing at read time. Past ~30 messages the two
lines separate and never meet again.

## 3. Break-even

Ingestion is paid on a cheap model; the saving is collected on the expensive
one, per recall. Price-weight them:

    break-even recalls = total ingestion cost / (tokens saved per recall × reader input price)

Worked example at 104 messages (paste-history alternative, ~3,100 tokens saved
per recall):

| Ingester → Reader | Ingestion of 104 msgs | Saved per recall | Break-even |
|---|---:|---:|---:|
| Haiku 4.5 → Opus 5 ($5/MTok in) | $0.27 | $0.0156 | ~17 recalls |
| Haiku 4.5 → Sonnet 5 ($3/MTok in) | $0.27 | $0.0093 | ~29 recalls |
| gpt-5-mini → Opus 5 | $0.07 | $0.0156 | ~5 recalls |
| gpt-5-mini → Sonnet 5 | $0.07 | $0.0093 | ~8 recalls |

Since history keeps growing, the saving per recall grows with it while the
ingestion cost per message stays flat, so break-even arrives sooner the longer
you use it. Note also that the "paste history" alternative stops being
possible at all once the history exceeds what you are willing to spend per
turn, which for daily use happens within weeks.

Against "no memory", the token math is trivially in Memry's favour (~700
tokens per recall, ~$0.0035 on Opus 5) and the real benefit is not tokens:

- **User time.** Re-stating one fact is a 30-60 word message. Assume 20-40 s
  of typing plus the message itself landing in the expensive model's context.
  Ten facts re-explained per week is minutes of your time and cents of tokens,
  every week, forever. This is an assumption, not a measurement.
- **Correctness.** The verbatim/RAG alternative served the superseded fact
  first. An agent acting on "lives in Munich" plans the wrong commute. Memry's
  reconciliation is what makes the recalled context trustworthy, and that is
  what the ingestion tokens buy.
- **Prefill time.** Fewer tokens per turn is faster time-to-first-token on the
  reader; a few thousand tokens is tens to hundreds of milliseconds. Real but
  minor.

## 4. When not to use it

- Single-session work with no future sessions: ingestion is pure cost.
- Short histories under ~30 messages where you could paste everything.
- No cheap-model budget at all: run Memry with no LLM key. Ingestion is then
  free (verbatim + hash embeddings + BM25), you keep search and the audit
  trail, and you lose reconciliation and distillation. That is still a real
  cross-session memory, just a dumber one.

## 5. Levers that change the ratio

Ordered by measured impact:

1. **Entity identity checks are the largest optional cost**: 24 of 62 calls
   (26% of input tokens). Batching the judgements per message, or skipping the
   LLM judge when the surface form and type match an existing entity exactly,
   would remove most of them.
2. **Prefer the deferred/MCP path** everywhere the client can tolerate a
   short delay; it halves ingestion by extracting per burst, not per message.
3. **Make the system prompts cacheable** on the chosen provider: either
   consolidate the ingestion prompts into one stable prefix above the
   provider's minimum, or accept full price and keep them short. Right now
   they are in the worst spot: too long to be cheap, too short to cache.
4. **Trim context rendering**: ~40% of a recall's tokens are formatting.
   Dropping the per-line date, or the type tag, or shortening the footer, is
   free savings on every recall on the expensive model.
5. **The coverage audit** (0.8 calls/msg, 7% of input) is a quality feature
   with a token price; it is already absent on the deferred path and could be
   config-gated on the synchronous one.

## 6. Reproduce

    python evals/cost_measure.py            # ingestion + read side, writes evals/cost_measure.json
    python evals/cost_measure.py --scale    # history growth to 104 messages, writes evals/cost_scaling.json

Prices quoted are Anthropic list prices as of the pricing reference available
when this was written (2026-08-17); check current price lists before quoting
them onward.
