# CodeSight 

**CodeSight** -- an AI code review pipeline that generates verified, evidence-backed review comments on real-world pull requests.

CodeSight doesn't just find issues -- it generates fixes, writes regression tests, runs them, and only posts comments that survive a multi-stage verification pipeline.

---

## Results

| Metric | Value |
|--------|-------|
| **Address Rate** | **58.3%** |
| **Precision** | **58.3%** |
| **Plausible Rate** | **41.7%** |
| **Verified Rate** | **100%** |
| **Cost per PR** | **~$0.20** |

> Evaluated on LLM-generated review comments across real merged PRs from [fastapi/fastapi](https://github.com/fastapi/fastapi). Address rate = percentage of AI comments that matched issues human reviewers also flagged and developers actually fixed. Plausible = novel finds on changed code not flagged by humans.

---

## Architecture

```
PR Input
  |
  v
+------------------+     +----------+
| GitHub API       |---->| Clone PR |
+------------------+     +----+-----+
                               |
                               v
                    +---------------------+
                    | Neo4j Code Graph    |
                    | (load or patch)     |
                    +----+----------------+
                         |
              +----------+----------+
              |                     |
              v                     v
    +------------------+   +------------------+
    | Static Analysis  |   | Investigator     |
    | (Ruff, Semgrep)  |   | Query graph for: |
    |                  |   | - callers        |
    |                  |   | - callees        |
    |                  |   | - blast radius   |
    |                  |   | - bug history    |
    +--------+---------+   +--------+---------+
             |                      |
             |         +------------+
             |         |
             |         v
             |  +-------------------------------+
             |  | Reviewer (Claude Sonnet)      |
             |  | Structured prompting:         |
             |  |  - premise / evidence /       |
             |  |    trigger / impact           |
             |  | + Pinecone RAG guidance       |
             |  +-------------------------------+
             |         |
             |         v
             |  +-------------------------------+
             |  | Actionability Filter (Haiku)  |
             |  | Drops vague/speculative       |
             |  | comments before verification  |
             |  +-------------------------------+
             |         |
             |         v
             |  +-------------------------------+
             |  | Verifier (Haiku)              |
             |  |  1. Generate minimal fix      |
             |  |  2. AST diff (real vs cosmetic)|
             |  |  3. Compile check             |
             |  |  4. Generate regression test  |
             |  |  5. Run test (pytest/jest)    |
             |  +-------------------------------+
             |         |
             |         v
             |  +-------------------------------+
             |  | Critic (Haiku + Pinecone)     |
             |  | Score 0.0-1.0 per comment     |
             |  | Pinecone: kill if 3+ historic |
             |  |   downvotes, boost if upvotes |
             |  +-------------------------------+
             |         |
             +----+----+
                  |
                  v
           +-------------+
           | Ranker       |
           | Dedup, sort, |
           | cap to top N |
           +------+------+
                  |
                  v
           +-------------+
           | Post to      |
           | GitHub PR    |
           +------+------+
                  |
                  v
           +------------------+
           | Feedback Loop    |
           | Record outcomes  |
           | --> Pinecone     |
           | (learning loop)  |
           +------------------+
```

---

## What Makes ARES Different

### 1. Verification-Backed Comments
Every comment goes through fix generation + compilation + test execution before posting. The verifier generates a minimal fix, checks if it compiles, writes a regression test, and runs it. Comments that fail verification are killed. **100% of posted comments are verified.**

### 2. Graph-Aware Review
ARES builds a full code graph (AST nodes + call edges + git metadata) in Neo4j. During review, it queries:
- **Blast radius**: How many functions transitively depend on the changed code?
- **Bug history**: How often has this function been involved in bug fixes?
- **Co-change patterns**: What files typically change together?

This context drives target prioritization -- high-risk, high-impact functions get reviewed first.

### 3. Structured Prompting
Inspired by Meta's semi-formal reasoning research (78% to 93% accuracy), the reviewer prompt requires structured evidence:
- **Premise**: What contract is being violated?
- **Evidence**: Which diff lines show it?
- **Trigger**: What input causes it?
- **Impact**: What breaks?

This eliminates speculative "consider doing X" comments that developers ignore.

### 4. Actionability Filter
Based on Atlassian's finding that actionability filtering is the single highest-ROI intervention (+15-20pp address rate), ARES classifies every comment as actionable or not before spending compute on verification. Vague, speculative, or stylistic comments are dropped.

### 5. Feedback Learning Loop (Pinecone RAG)
Historical review outcomes are stored as embeddings in Pinecone:
- **Upvotes**: Comments developers acted on -- used as tone/focus guidance for the reviewer
- **Downvotes**: Comments developers ignored -- used to kill similar future comments (3+ downvotes = hard kill)

The system learns from every reviewed PR.

### 6. Multi-Review Aggregation (Optional)
Run the reviewer N times with temperature variation, cluster results by semantic similarity, keep only findings that 2+ independent runs agree on. Based on SWR-Bench research showing +43% F1 improvement.

### 7. Cost-Optimized Pipeline
| Component | Model | Why |
|-----------|-------|-----|
| Reviewer | Claude Sonnet | Core intelligence -- needs deep reasoning |
| Verifier | Claude Haiku | Mechanical code gen -- fix + test generation |
| Critic | Claude Haiku | Classification task -- score 0-1 |
| Actionability | Claude Haiku | Binary classification |

Token-optimized: function source trimmed to diff-adjacent lines, caller/callee context reduced to signatures only. **~$0.20 per PR** vs industry tools at $1-5+.

---

## Quick Start

### Prerequisites
- Python 3.11+
- Neo4j Aura (free tier works) or local Neo4j
- Pinecone (free tier works)
- Anthropic API key
- GitHub token

### Environment Variables
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export GITHUB_TOKEN="ghp_..."
export PINECONE_API_KEY="pcsk_..."
export NEO4J_URI="neo4j+s://xxxxx.databases.neo4j.io"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="..."
```

### Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Build the Code Graph
```bash
python scripts/build_graph.py --repo fastapi/fastapi --branch master --clone-depth 100
```

### Seed Pinecone with Historical Feedback
```bash
python scripts/seed_pinecone.py --repo fastapi/fastapi --max-prs 200
```

### Run Evaluation
```bash
python scripts/eval_comment_sample.py \
    --repo fastapi/fastapi \
    --target-comments 50 \
    --max-inspected-prs 200 \
    --min-human-comments 1 \
    --base-branch master
```

---

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ARES_MODEL` | `claude-sonnet-4-6` | Reviewer model |
| `ARES_LIGHTWEIGHT_MODEL` | `claude-haiku-4-5-20251001` | Verifier/Critic model |
| `ARES_MAX_COMMENTS` | `3` | Max comments per PR |
| `ARES_REVIEW_MAX_PASSES` | `1` | Review refinement passes |
| `ARES_REVIEW_AGGREGATION_RUNS` | `1` | Multi-review runs (set 2-3 to enable) |
| `ARES_ACTIONABILITY_FILTER` | `1` | Enable actionability filter |
| `ARES_PINECONE_INDEX` | `ares-comments` | Pinecone index name |
| `ARES_PINECONE_NAMESPACE` | `default` | Pinecone namespace |

---

## Project Structure

```
ares/
  agents/
    _llm.py             # LLM adapter (Anthropic + OpenAI, temperature support)
    reviewer.py          # Structured prompting, Pinecone RAG, multi-review
    critic.py            # Heuristic + LLM scoring, actionability filter, Pinecone boost/kill
    verifier.py          # Fix generation, AST diff, compilation, test execution
    investigator.py      # Target selection, context assembly
  graph/
    parser.py            # Tree-sitter + AST parsing, progress logging
    classifier.py        # Risk classification, parallel git metadata
    indexer.py           # Build/patch/load graph with Neo4j
    query.py             # Graph traversal, target building, source trimming
  integrations/
    github_client.py     # PR data, cloning, review threads, ground truth
    neo4j_client.py      # Graph persistence, Cypher queries, Aura compatible
    pinecone_client.py   # Feedback storage, similarity search, no fallbacks
  feedback/
    collector.py         # Record outcomes, upsert to Pinecone
    learner.py           # Strategy adaptation from feedback
    strategy.py          # Configurable review parameters
  evaluate.py            # Address rate calculation, human comment matching
  pipeline.py            # Orchestrator -- review_pr, batch, aggregation
  ranker/ranker.py       # Priority sorting, dedup, static+LLM merge
  static_analysis/       # Ruff + Semgrep integration
scripts/
  build_graph.py         # One-time graph indexing into Neo4j
  seed_pinecone.py       # Bootstrap Pinecone with historical PR feedback
  eval_comment_sample.py # Evaluation harness with progress tracking
tests/                   # Unit tests for all components
```

---

## Research References

| Technique | Source | Impact |
|-----------|--------|--------|
| Multi-Review Aggregation | SWR-Bench (2025) | +43% F1 |
| Actionability Filtering | Atlassian RovoDev (2026) | +15-20pp address rate |
| Structured Prompting | Meta Agentic Code Reasoning (2026) | 78% to 93% accuracy |
| Feedback RAG | CodeRabbit architecture | #1 on Martian Bench |
| CRScore evaluation | NAACL 2025 | 0.54 Spearman with human judgment |
| PR-Agent prompt design | Qodo (open source) | Industry standard patterns |

