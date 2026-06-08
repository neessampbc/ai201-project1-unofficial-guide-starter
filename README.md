# The Unofficial Guide — Project 1

A Retrieval-Augmented Generation (RAG) system that answers plain-language questions about **CUNY Hunter College professors** using student reviews, with grounded answers and source citations.

> **Data note:** The review corpus is **synthetic** — fictional professor names paired with real Hunter departments/course codes, created for this educational project. Every file is labeled as synthetic so no fabricated opinions are attributed to real people. The pipeline is identical to one run on real collected reviews.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then put your Groq API key in .env

python ingest.py        # inspect chunks (no key needed)
python embed_store.py   # build the ChromaDB index (no key needed)
python query.py         # retrieval smoke test (no key needed)
python evaluate.py      # run the 5 eval questions end-to-end (needs key)
python app.py           # launch Gradio UI at http://localhost:7860 (needs key)
```

**Pipeline:** `documents/*.txt` → `ingest.py` (load + clean + chunk) → `embed_store.py` (all-MiniLM-L6-v2 + ChromaDB) → `query.py` (`retrieve` + grounded `ask`) → `app.py` (Gradio).

---

## Domain

Student reviews of professors at CUNY Hunter College. Official course catalogs list a course's title and credits but not the things that actually decide your semester: whether exams come from the slides or the textbook, how heavy the weekly workload is, whether the professor curves, or whether attendance is quietly part of the grade. That experiential knowledge lives in scattered reviews, group chats, and word of mouth. This system makes it searchable and answerable with citations.

---

## Document Sources

12 plain-text files in `documents/`, one professor per file, 5-6 student reviews each (Rating / Difficulty / Would-take-again plus free-text opinion). Six departments, a spread of sentiment, so the corpus answers a range of questions.

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Prof. Rivera — CSCI 135 (CS) | Synthetic reviews (.txt) | documents/rivera_csci135.txt |
| 2 | Prof. Zhang — CSCI 235 (CS) | Synthetic reviews (.txt) | documents/zhang_csci235.txt |
| 3 | Prof. Chen — MATH 155 (Math) | Synthetic reviews (.txt) | documents/chen_math155.txt |
| 4 | Prof. Petrova — MATH 160 (Stats) | Synthetic reviews (.txt) | documents/petrova_math160.txt |
| 5 | Prof. Alvarez — ENGL 120 (English) | Synthetic reviews (.txt) | documents/alvarez_engl120.txt |
| 6 | Prof. Okafor — PSYCH 100 (Psych) | Synthetic reviews (.txt) | documents/okafor_psych100.txt |
| 7 | Prof. Kim — BIOL 100 (Biology) | Synthetic reviews (.txt) | documents/kim_biol100.txt |
| 8 | Prof. Bennett — ECO 200 (Econ) | Synthetic reviews (.txt) | documents/bennett_eco200.txt |
| 9 | Prof. O'Brien — HIST 151 (History) | Synthetic reviews (.txt) | documents/obrien_hist151.txt |
| 10 | Prof. Goldstein — PHILO 101 (Philosophy) | Synthetic reviews (.txt) | documents/goldstein_philo101.txt |
| 11 | Prof. Hassan — SOC 101 (Sociology) | Synthetic reviews (.txt) | documents/hassan_soc101.txt |
| 12 | Prof. Lang — ANTHC 101 (Anthropology) | Synthetic reviews (.txt) | documents/lang_anthc101.txt |

**Ingestion/cleaning:** `ingest.py` reads each file, unescapes HTML entities, strips any stray tags, normalizes whitespace (preserving paragraph breaks), drops the header and synthetic-data `NOTE:` line, and splits the body into individual reviews on blank lines.

---

## Chunking Strategy

**Chunk size:** One review per chunk (~230-350 chars in practice), capped at 600 chars before windowing.

**Overlap:** 80 characters, applied only to reviews that exceed the 600-char cap.

**Why these choices fit your documents:** Each review is already a complete, self-contained opinion, so the review is the natural retrieval unit. A fixed "every 500 characters" split would cut a reviewer's point in half and merge two unrelated opinions, hurting precision. Each chunk is prefixed with `[Professor <Name>, <Course>]` so the professor/course is part of the embedded text and the chunk stands alone for attribution. Overlap only matters for the rare long review where a fact sits near a forced split.

**Final chunk count:** 64 chunks across 12 documents (min 231 / avg 298 / max 347 chars) — comfortably inside the 50-2,000 guidance.

### Sample chunks (5, each with source)

1. **rivera_csci135.txt** — `[Professor Daniel Rivera, CSCI 135] Rating: 4.5/5 | Difficulty: 3/5 | Would take again: Yes — Rivera's exams come almost entirely from his lecture slides, not the Savitch textbook. If you go to every lecture and rewrite the slide code by hand, you will be fine. He curves the midterm but not the final...`
2. **chen_math155.txt** — `[Professor Wei Chen, MATH 155] Rating: 4/5 | Difficulty: 4/5 | Would take again: Yes — Hard but fair. Professor Chen does not curve, but the homework counts for 25% of your grade, so if you do all of it you build a cushion. Partial credit on exams is generous as long as you show your steps.`
3. **alvarez_engl120.txt** — `[Professor Maria Alvarez, ENGL 120] Rating: 4.5/5 | Difficulty: 3/5 | Would take again: Yes — I'd recommend Alvarez to anyone. Her class discussions are genuinely engaging... Just know that she is a tough grader on essays.`
4. **okafor_psych100.txt** — `[Professor Grace Okafor, PSYCH 100] Rating: 3.5/5 | Difficulty: 2/5 | Would take again: Yes — Fair and predictable. Exams are 50 multiple-choice questions each. There's one APA-style paper worth 15% that trips people up...`
5. **zhang_csci235.txt** — `[Professor Wenjie Zhang, CSCI 235] Rating: 4/5 | Difficulty: 4/5 | Would take again: Yes — A lot of students call her "Z"... Z is demanding but she genuinely cares — her office hours saved my hash table project.`

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`, run locally with L2-normalized embeddings stored in ChromaDB (cosine distance). Free, fast, no API key or rate limits — ideal for short review text.

**Production tradeoff reflection:** If deploying for real users with cost off the table, I'd weigh:
- **Accuracy on domain text** — a larger model (`bge-large-en-v1.5`, OpenAI `text-embedding-3-large`) separates near-duplicate opinions better than 384-dim MiniLM.
- **Context length** — MiniLM truncates around 256 tokens; fine for reviews, limiting if documents grew into long guides.
- **Multilingual support** — Hunter is highly multilingual; reviews in Spanish/Chinese would need a multilingual model (`paraphrase-multilingual-MiniLM`, Cohere multilingual).
- **Latency & privacy** — API embeddings add network latency and send student data to a third party; local models avoid both.

---

## Retrieval Test Results

Three evaluation queries run through `retrieve(query, k=4)` (output from `python query.py`). All top results are below 0.5 cosine distance and come from the correct source file.

**Query 1 — "What do students say about Professor Rivera's exams in CSCI 135?"**
```
[0.271] rivera_csci135.txt  "...Rivera's exams come almost entirely from his lecture slides..."
[0.307] rivera_csci135.txt  "...The labs are graded harshly by the TAs..."
[0.347] rivera_csci135.txt  "...Attendance matters way more than the readings..."
[0.366] rivera_csci135.txt  "...If you have any programming background this class will feel easy..."
```
*Why relevant:* All four chunks are Rivera/CSCI 135 reviews, and the top hit (0.271) directly answers the exam question ("exams come almost entirely from his lecture slides"). The professor+course prefix on each chunk is why every result is correctly scoped to Rivera rather than another CS professor.

**Query 2 — "Is Professor Chen's MATH 155 a heavy workload?"**
```
[0.311] chen_math155.txt  "...not as scary as people say if you do the homework..."
[0.332] chen_math155.txt  "...Chen's lectures are extremely organized..."
[0.365] chen_math155.txt  "...Chen assigns WebAssign homework every single week..."
[0.388] chen_math155.txt  "...Hard but fair. Professor Chen does not curve..."
```
*Why relevant:* Every chunk is from Chen/MATH 155 and the set captures both sides of the workload question — the WebAssign-heavy chunk (0.365) and the "consistent, not crushing" chunk (0.311) — which is exactly what lets the generated answer summarize the disagreement honestly.

**Query 3 — "Do students recommend Professor Alvarez for ENGL 120?"**
```
[0.279] alvarez_engl120.txt  "...I'd recommend Alvarez to anyone..."
[0.373] alvarez_engl120.txt  "...Be ready to read. The reading load is heavier..."
[0.448] alvarez_engl120.txt  "...Alvarez changed how I write..."
[0.451] alvarez_engl120.txt  "...Workload is real but the class is worth it..."
```

---

## Grounded Generation

**System prompt grounding instruction:** The model is told to answer using ONLY the `CONTEXT` block, with explicit rules: base every claim on the context, use no outside knowledge, do not invent professors/courses/ratings, summarize disagreement, and — critically — *"If the CONTEXT does not contain enough information to answer the question, reply with exactly: 'I don't have enough information on that.'"* Temperature is set to 0.1 to minimize improvisation. (See `SYSTEM_PROMPT` in `query.py`.)

**Structural grounding (not just the prompt):**
- Retrieved chunks beyond a cosine distance of `0.9` are dropped before generation; if nothing survives, the system returns the refusal string without ever calling the LLM.
- **Source attribution is programmatic**, not left to the model: after generation, `ask()` collects the `source` metadata of the chunks that were actually passed as context and returns them as `sources`. The model cannot fabricate or omit a citation.

---

## Example Responses

Actual output from `ask()` (answer text + programmatic `sources`).

**Example 1 — grounded answer with source attribution**
> **Q:** What do students say about Professor Rivera's exams in CSCI 135?
> **A:** According to the students, Professor Rivera's exams in CSCI 135 come almost entirely from his lecture slides. To do well, students recommend attending every lecture, rewriting the slide code by hand, and studying the slides and lecture examples. The midterm is curved, but the final is not. One student noted that half of the exam questions are similar to in-class examples with changed numbers.
> **Retrieved from:** rivera_csci135.txt

**Example 2 — grounded answer with source attribution**
> **Q:** How are exams graded in Professor Okafor's PSYCH 100?
> **A:** In Professor Okafor's PSYCH 100, exams are multiple-choice (50 questions each), pulled from the textbook and lecture, with no essays on the tests.
> **Retrieved from:** okafor_psych100.txt

**Example 3 — out-of-scope query (refusal)**
> **Q:** Which dining hall at Hunter has the best food?
> **A:** I don't have enough information on that.
> **Retrieved from:** (no sources — question outside the document set)

---

## Query Interface

A Gradio web UI (`python app.py`, served at http://localhost:7860).

**Input field:** `Your question` — a single textbox for a plain-language question. Submit with the **Ask** button or the Enter key. Five example questions are provided as one-click presets.

**Output fields:**
- `Answer` — the grounded response generated from retrieved reviews.
- `Retrieved from` — the bulleted list of source documents the answer drew from (or a note that the question fell outside the document set).

**Sample interaction transcript:**
```
Your question:  Is Professor Chen's MATH 155 a heavy workload?

Answer:         According to students, Professor Chen's MATH 155 has a consistent
                workload. WebAssign homework is assigned weekly with 30-40 problems
                due each Sunday night, which several reviewers called a heavy load.
                Most agreed it is manageable if you keep up and that the homework
                mirrors the exams.

Retrieved from: • chen_math155.txt
```

---

## Evaluation Report

Run with `python evaluate.py`. Retrieval quality judged by whether top-k chunks are on-topic and from the right source (distances shown); response accuracy judged against the expected answers in `planning.md`.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What do students say about Professor Rivera's exams in CSCI 135? | Exams from lecture slides; attendance/redoing examples matters; midterm curved, final not. | Exams come from slides; attend lectures and rewrite slide code; midterm curved, final not; questions mirror in-class examples. | Relevant (top dist 0.271, all from rivera) | Accurate |
| 2 | Is Professor Chen's MATH 155 a heavy workload? | Yes — weekly WebAssign (30-40 problems), ~25% of grade, mirrors exams; hard but fair. | Consistent but heavy weekly WebAssign (30-40 problems due Sundays); manageable if you keep up; helps with exams. | Relevant (top dist 0.311, all from chen) | Accurate |
| 3 | Do students recommend Professor Alvarez for ENGL 120? | Generally yes; engaging, improves writing, but tough essay grader and heavy reading. | Most recommend her (3 of 4 positive); praised for improving writing; one negative cites heavy reading load. | Relevant (top dist 0.279, all from alvarez) | Accurate (slightly under-states the "tough grader" angle) |
| 4 | How are exams graded in Professor Okafor's PSYCH 100? | 3 non-cumulative multiple-choice exams + attendance quizzes + one APA paper; easy if you show up. | Multiple-choice exams (50 questions each) from textbook/lecture; no essays on tests. | Relevant (top dist 0.194, all from okafor) | Partially accurate (correct on MC exams; omitted attendance quizzes and the APA paper) |
| 5 | What are Professor Z's projects like in CSCI 235? | "Z" = Prof. Zhang; four large C++ projects, strict style/memory grading, weighted above exams, no late work. | Four big C++ projects worth more than exams; linked list/hash table 20+ hrs; no late submissions. | **Partially relevant** (top dist 0.492; an off-topic rivera_csci135 chunk ranked #3 at 0.547 and leaked into sources) | Accurate text, but **source attribution wrong** (rivera_csci135.txt cited) |

**Retrieval quality:** Relevant (Q1-Q4) / Partially relevant (Q5)
**Response accuracy:** Accurate (Q1, Q2, Q3) / Partially accurate (Q4) / Accurate-with-bad-citation (Q5)

---

## Failure Case Analysis

**Question that failed:** "What are Professor Z's projects like in CSCI 235?"

**What the system returned:** The *answer text* was correct (four large C++ projects, graded strictly, no late work), but the `sources` list was **`['zhang_csci235.txt', 'rivera_csci135.txt']`** — it cited Rivera, an unrelated CS professor whose reviews never mention CSCI 235's projects.

**Root cause (tied to a specific pipeline stage):** This is a **retrieval / embedding** failure. Students nickname Professor Zhang "Z", but the indexed chunks and metadata use the formal surname "Zhang". Querying with the nickname "Professor Z" has weak lexical/semantic overlap with the embedded text, so every distance rose into the 0.49-0.56 range (vs. 0.19-0.39 for the other questions). With the correct chunks only marginally ahead, a generic CSCI chunk from `rivera_csci135.txt` (distance 0.547, "Best intro professor in the CS department...") slipped into the top-4. Because `ask()` builds the citation list from whatever chunks pass the distance filter, that off-topic chunk polluted the sources even though the answer itself stayed correct.

**What you would change to fix it:** (1) Add a nickname/alias map so "Z" expands to "Zhang" at query time, or store known aliases in the chunk text/metadata so they get embedded. (2) Tighten the distance filter (e.g., drop chunks above ~0.45, or keep only chunks within a margin of the best score) so a weak straggler can't enter the cited sources. (3) Optionally add metadata filtering by course code so a "CSCI 235" question can't pull a CSCI 135 chunk.

---

## Spec Reflection

**One way the spec helped you during implementation:** Deciding the chunking strategy in `planning.md` *before* writing code — "one review per chunk, prefixed with professor/course" — directly shaped `ingest.py` and turned out to be the single biggest reason retrieval scored so well (distances of 0.19-0.39 on four of five questions). Because the unit of meaning was settled up front, I never had to retrofit the chunker after seeing bad retrieval; the chunks were self-contained from the first run.

**One way your implementation diverged from the spec, and why:** The spec only described retrieving top-k and generating. During implementation I added a **distance threshold + refusal-before-LLM** path that wasn't in the original plan, because testing an out-of-scope question ("best dining hall") showed the model would otherwise try to answer from whatever low-relevance chunks came back. Adding the filter made grounding a structural guarantee rather than something I only hoped the prompt enforced. The Q5 failure later showed the threshold (0.9) is still too loose for citations — a documented next step.

---

## AI Usage

**Instance 1**
- *What I gave the AI:* The `Chunking Strategy` and `Documents` sections of `planning.md`, plus the file format (header line, blank-line-separated reviews), and asked it to implement the ingestion + chunking in `ingest.py`.
- *What it produced:* A loader and a chunker that split on review boundaries, prefixed each chunk with professor/course, windowed long reviews with overlap, and attached `{source, professor, course, chunk_index}` metadata.
- *What I changed or overrode:* Verified against the printed 5-sample output and the 64-chunk count; confirmed the header regex correctly parsed course codes and that the synthetic-data `NOTE:` line was excluded from chunks so it never pollutes retrieval.

**Instance 2**
- *What I gave the AI:* The grounding requirement (answer only from retrieved context, exact refusal string) and the desired return shape `{answer, sources, chunks, distances}`, and asked it to implement `ask()` in `query.py`.
- *What it produced:* A function that formats retrieved chunks into a `CONTEXT` block, calls Groq `llama-3.3-70b-versatile` with a strict system prompt, and returns the answer.
- *What I changed or overrode:* Directed it to make source attribution **programmatic** (built from chunk metadata) instead of asking the LLM to cite sources, and added a distance filter that returns the refusal *before* calling the LLM when no chunk is relevant — so grounding is enforced by the pipeline, not just suggested in the prompt.
