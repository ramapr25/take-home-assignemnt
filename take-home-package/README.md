# Larkspur repair-note extraction

This pulls six fields out of each of the 500 technician notes and writes them to `predictions.jsonl`. It also scores the result against the 50 labeled notes and reports what the run cost. To set it up and run it, see [how_to_run.md](how_to_run.md). In short: `pip install -r requirements.txt`, put `ANTHROPIC_API_KEY` in `.env`, then `python run.py`. The libraries and services used are listed in [tech_stack_used.md](tech_stack_used.md).

**Where things live:**

| File | Contents |
|---|---|
| `run.py` | Runs the pipeline. |
| `larkspur/extractors.py` | All the regex and phrase rules. |
| `larkspur/llm_extractor.py` | The prompt, the Claude call and the fallback chain. |
| `larkspur/pipeline.py` | Ties the steps together. |
| `larkspur/evaluation.py` | Scoring. |
| `larkspur/config.py` | Paths, models and prices. |

**Result.** Every field scored 50/50 on the labeled sample. Across all 500 notes, Claude's answers matched an independent rule-based pass on 1,499 of 1,500 values. The one disagreement is a model error, described under [the hardest cases](#the-three-hardest-cases). The measured cost is **$1.35 per 1,000 notes** with Claude Haiku 4.5.

## Approach

Before writing any code, I read through the notes. They are heavily templated, with about 600 distinct sentence shapes. That decided the split: plain code handles the fields that follow a pattern, and the model handles the fields that depend on meaning.

| Field | How | Why |
|---|---|---|
| `model_number` | Regex | It has a fixed shape (`ABC-1234[X]`). The tricky part is that part numbers like `LPN-5648` fit the same shape, so I remove part and stock references and any "customer also asked me to look at…" clause first. The field is null when the note says the plate was unreadable or missing. |
| `visit_date` | Regex + `dateutil` | Seven date formats. "Today" and "yesterday" are resolved against the UTC date of `submitted_at`. |
| `follow_up_needed` | Phrase list | True only for a scheduled return visit or a part on order. "Call us if it happens again" is false, as the schema says. |
| `appliance_type`, `problem_category`, `part_replaced` | Claude Haiku 4.5 | These need judgment: the customer's symptom rather than the diagnosis, `error_code` only when a code is the *only* complaint, a part installed versus only ordered, and ignoring a second appliance. The model returns structured output, using a JSON schema built from the allowed values in `schema.json`, so it can't return an invalid label. Temperature is 0 and failed calls are retried. |

I also wrote rule-based versions of the three model fields. They serve three purposes:
- **A cross-check.** Every run reports where the model and the rules disagree, which covers all 500 notes, not just the 50 labels.
- **The last step of the fallback chain** (below).
- **A free dry-run mode** (`--dry-run`).

**Fallback chain.** For each note, the pipeline tries Haiku 4.5 first. If that fails after retries, it tries a fallback model (Haiku 5.5 by default), and only then uses the rule answer. A run never stops because of one bad note, and every note gets an answer.
- If the primary model can't be used at all (unknown model ID, or no access with this key), the pipeline stops calling it and sends the rest of the run straight to the fallback.
- A rejected API key stops the run immediately, because every model would fail the same way.
- `outputs/run_stats.json` records how many notes each model or the rules answered, with token use and cost for each model.
- Both models are set in `.env` (`LARKSPUR_MODEL`, `LARKSPUR_FALLBACK_MODEL`), so you can switch them without touching the code. `LARKSPUR_FALLBACK_MODEL=none` turns the fallback off.

I tested this chain offline against a simulated API: a missing model, repeated server errors, both models down, and a bad key. In the real 500-note run it never triggered, because every Haiku 4.5 call succeeded.

## Accuracy on the labeled sample (n = 50)

| Field | Rules only | Rules + Haiku 4.5 |
|---|---|---|
| appliance_type | 50/50 | 50/50 |
| model_number | 50/50 | 50/50 (same rules) |
| problem_category | 50/50 | 50/50 |
| part_replaced | 50/50 | 50/50 |
| visit_date | 50/50 | 50/50 (same rules) |
| follow_up_needed | 50/50 | 50/50 (same rules) |
| **All six fields correct** | **50/50** | **50/50** |

Across all 500 notes, the model and the rules agree on 1,499 of the 1,500 values for the three model fields. These are two methods that work very differently, so their agreement is my best evidence that the unlabeled notes are right too. The one disagreement, N0467, is not in the labeled sample, and the model is the one that got it wrong. The check caught an error that the 50 labels could not.

What 50 examples can't tell you:
- **The true accuracy could still be about 93%.** That is the lower end of the 95% confidence interval for 50 out of 50, and a single error moves a field by 2 points.
- **The sample has no `error_code` notes**, while the full set has about 29. That label has no measured accuracy.
- **The sample is not a clean held-out test.** I wrote the rules by looking at all 500 notes and then checked them against the labels. I never added a rule to fix a specific labeled note, but a score on data you have seen is optimistic.

## Cost per 1,000 notes

The token counts are measured: `run.py` adds up the token usage the API reports for every call and writes it to `outputs/run_stats.json`. The full run made 500 calls, used 590,506 input and 16,474 output tokens, and cost **$0.67**. That averages 1,181 input and 33 output tokens per note. Most of the input is the system prompt and the output schema, which are the same for every note.

```
cost per 1,000 notes = 1,000 × (1,181 × $1.00 + 33 × $5.00) / 1,000,000
                     = $1.18 + $0.17 = $1.35          (Haiku 4.5, $1 / $5 per million tokens)
```

The rule-based fields cost nothing. There are two cheaper options I haven't measured:
- **The Message Batches API** halves the price, to about $0.67 per 1,000 notes.
- **Haiku 5.5** costs $0.10 / $0.50 per million tokens, which would be roughly **$0.14 per 1,000 notes**. That figure assumes the same token counts. Haiku 5.5 uses a different tokenizer and thinks by default, so the code turns thinking off for it. It is also the fallback model, so fallback calls are *cheaper* than primary ones and the fallback never raises the cost. I haven't run that path at scale.

## The three hardest cases

The final run had no mismatches against the labels and one disagreement between the model and the rules. That disagreement is the only observed error, so it comes first. The other two are the cases that were hardest to get right, and they are where I would look first if real notes start to drift from the templates:

1. **A part installed, plus a future visit that sounds like it isn't.** N0467: *"Replaced control board, tested ok. Need a second tech for the install, booked for next week."* The right answer is part = `control board` and follow-up = true. Haiku 4.5 answered `none` in two separate full runs, so the mistake is consistent, not random. The word "install" seems to make it think the board hasn't been fitted yet. The rules get it right. N0308 (*"Replaced door gasket, tested ok. Backordered, ETA 2 wks"*) sets the same trap, and the model gets that one right. I haven't changed the prompt for N0467, because tuning it to one known note would make the score less trustworthy. In production, this is exactly the kind of note the disagreement check would send to a person.
2. **A symptom plus a display code.** N0039: *"nothing happens when you press start, display also shows F21"*. The right label is `wont_start`, not `error_code`, because the code is not the only complaint.
3. **The repair confused with the symptom.** N0310: *"dead, no power … Actual fault was the door latch, replaced it"*. The problem is `wont_start` (what the customer reported), and the part is `door latch`.

## Assumptions and ambiguous cases

- **The year is 2026** when a date doesn't give one, as the schema states.
- **"Today" and "yesterday" use the UTC date.** All notes were submitted between 15:00 and 23:59 UTC, which is the same calendar day in every US time zone.
- **N0308-style notes** say a part was replaced and also that something is on order, without saying what. I followed the definitions literally: the installed part counts, and "on order" means follow-up is true.

## What I left out, and what I'd do next

I deliberately left out a few things to stay within the time limit:
- a held-out test split;
- a human-review queue;
- batching and prompt caching;
- testing on notes that don't follow the templates.

With more time, in priority order:
1. **Label more notes**, especially `error_code` notes and messier real-world ones, then split them into dev and test sets.
2. **Make the rules and the model check each other in production.** Accept a note automatically when they agree, and send disagreements to a person or a stronger model.
3. **Move bulk runs to the Batches API**, and check whether the system prompt is long enough to cache.
4. **Measure Haiku 5.5 properly**: run it on the labeled set, repeat the run to check consistency, then decide whether the 10× saving is worth switching.

## How I used AI

I built this with Claude Code (Claude Opus 5.5) as a pair programmer. It helped me:
- group the 500 notes into templates to see the patterns;
- draft the regexes and the system prompt;
- write the evaluation script;
- review the finished package against the brief.

At run time, the extraction model is Claude Haiku 4.5, with Claude Haiku 5.5 as the fallback.

How I checked the output:
- **Against the labels.** Every version was scored on the 50 labeled notes, and `outputs/errors.jsonl` lists each mismatch with its note text.
- **Two methods against each other.** The model was compared with the rules on all 500 notes, and `outputs/disagreements.jsonl` lists every disagreement.
- **The deliverable's format.** I validated `predictions.jsonl` against `schema.json`: all 500 notes are there, every value is allowed, and nulls and booleans are typed correctly.
- **By hand.** I read a random sample of predictions next to their notes, covering error-code notes, part numbers that look like model numbers, unreadable plates, relative dates and parts on order.

## Time spent

About **3.5 hours** in total, slightly over the suggested 3.
