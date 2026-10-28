# What Memry costs and what it saves

Why run a memory layer at all? Memry spends cheap-model tokens once, at ingestion, to make
every later recall cheaper on the expensive model and, more importantly, correct across
sessions. The break-even is real and it is not at message one.

`evals/cost_measure.py` measures it on your own setup: the real `MemoryStore` pipeline
driven by a scripted multi-session conversation (duplicates, one contradiction, recurring
entities), with a fake LLM that answers every prompt type plausibly and counts calls and
tokens (`len // 4`, the same estimator Memry uses). Token counts are estimates; the call
counts are exact. Re-run it after changing the pipeline, and price the result with your
own models' list prices.

## 1. Ingestion: what a saved message costs

A message saved with `store.add(infer=True)` goes through these phases, each a text-model
call or a decision question:

| Phase | What it does |
|---|---|
| Extraction | pulls out the facts, their type and importance, and the entities they name |
| Entity identity checks | decides whether a name is an entity the store already has |
| Reconciliation | per candidate fact: new, the same, more detail, a change or a correction |
| Coverage audit | names input details no stored fact captured |

Two things matter more than the total:

- **Most of the input is fixed system prompt.** Your text plus the retrieved neighbours is
  the small, variable part. Whether the fixed part is cheap depends on prompt caching, and
  Memry's system prompts are shorter than the cacheable minimum of several providers, so on
  the models people usually pick for ingestion, assume the full price.
- **The MCP path is cheaper.** `save_memories` uses `add_deferred`; the background worker
  groups a session's saves into one extraction call instead of one per message.

Embeddings: the message plus each stored fact, batched; free with the default hash
embedder, one HTTP call per batch with OpenAI.

The cost is paid whether or not the memory is ever read.

Wall clock: the calls are sequential. On the MCP path they run in the background worker
and the agent's turn does not wait; on the Python `add()` path they block.

## 2. Read side: what a recall costs and what it replaces

`reconstruct_context(query, token_budget=...)` returns a bounded context, locally and in
milliseconds with hash embeddings (a remote embedding call adds its own latency). Part of
it is formatting (header, footer, the per-line date and type); that overhead is a lever,
see section 5.

What that replaces depends on what you would otherwise do:

| Alternative | Tokens injected per recall | Notes |
|---|---|---|
| Nothing (no cross-session memory) | none | The information is simply gone next session. The cost is re-explaining, or wrong assumptions. |
| Paste the whole history | grows with the history | Real chats with assistant turns are several times larger than the user's messages. |
| Verbatim RAG over raw messages (Memry with no LLM) | small | Cheap to build, but it can serve a stale fact first: a message saying someone moved does not retire the older one. |
| Memry distilled context | capped by the budget | A changed fact supersedes the old one, and the context stops growing once it reaches the budget. |

With a short history Memry saves nothing at read time. As the history grows, pasting it
keeps growing while Memry's context stays at its cap, and the two never meet again.

## 3. Break-even

Ingestion is paid on a cheap model; the saving is collected on the expensive one, per
recall. Price-weight them:

    break-even recalls = total ingestion cost / (tokens saved per recall × reader input price)

Since history keeps growing, the saving per recall grows with it while the ingestion cost
per message stays flat, so break-even arrives sooner the longer you use it. Note also that
the "paste history" alternative stops being possible at all once the history exceeds what
you are willing to spend per turn, which for daily use happens within weeks.

Against "no memory", the token math is trivially in Memry's favour and the real benefit is
not tokens:

- **User time.** Re-stating a fact costs a message of your time plus that message landing
  in the expensive model's context, every week, forever.
- **Correctness.** A verbatim or RAG alternative can serve a superseded fact first. An
  agent acting on an old address plans the wrong commute. Memry's reconciliation is what
  makes the recalled context trustworthy, and that is what the ingestion tokens buy.
- **Prefill time.** Fewer tokens per turn is a faster time-to-first-token on the reader.
  Real but minor.

## 4. When not to use it

- Single-session work with no future sessions: ingestion is pure cost.
- Short histories you could paste whole.
- No cheap-model budget at all: run Memry with no LLM key. Ingestion is then free
  (verbatim + hash embeddings + BM25), you keep search and the audit trail, and you lose
  reconciliation and distillation. That is still a real cross-session memory, just a
  dumber one.

## 5. Levers that change the ratio

1. **Entity identity checks are the largest optional cost.** Batching the judgements per
   message, or skipping the judge when the surface form and type match an existing entity
   exactly, would remove most of them.
2. **Prefer the deferred/MCP path** everywhere the client can tolerate a short delay; it
   extracts per burst, not per message.
3. **Make the system prompts cacheable** on the chosen provider: either consolidate the
   ingestion prompts into one stable prefix above the provider's minimum, or accept full
   price and keep them short.
4. **Trim context rendering**: dropping the per-line date, or the type tag, or shortening
   the footer, is free savings on every recall on the expensive model.
5. **The coverage audit** is a quality feature with a token price; it runs on both paths
   (once per group on the deferred one) and could be config-gated.

## 6. Reproduce

    python evals/cost_measure.py            # ingestion + read side, writes evals/cost_measure.json
    python evals/cost_measure.py --scale    # history growth, writes evals/cost_scaling.json

Check current price lists before turning token counts into money.
