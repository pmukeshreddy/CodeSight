# ARES

ARES is an agentic code review system built around a simple premise: reduce noisy model inputs before generating comments, then ship every surviving comment with a validated fix.

## What is included

- Repository indexing with AST extraction and a lightweight knowledge graph
- Risk classification and git-history enrichment
- PR-time graph queries and focused reviewer context assembly
- Static-analysis fast path for ruff and semgrep
- Reviewer, verifier, critic, and ranker stages
- GitHub and Pinecone integration wrappers
- A CLI entrypoint for indexing, reviewing, and evaluation

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ares.pipeline index --repo /path/to/local/repo
python -m ares.pipeline review --repo owner/repo --pr 123 --target-dir /tmp/ares-review
python -m ares.pipeline evaluate --repo owner/repo --prs 123,124 --target-dir /tmp/ares-review
```

## Environment

Set the keys you intend to use:

- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
- `GITHUB_TOKEN`
- `PINECONE_API_KEY`
- `ARES_MODEL`
- `ARES_PROVIDER`
- `ARES_PINECONE_INDEX`

The deterministic parts of the system can run without API keys. LLM-backed stages will return no comments when no provider is configured.

`evaluate` will fetch GitHub review comments for each PR and label them as addressed when later commits modify the same file/line span, so `GITHUB_TOKEN` is required unless you supply `human_comments` in fixture data.

## Historical eval sampling

For a 100-comment historical evaluation run, the sampler will auto-raise the per-PR cap to `10` if `ARES_MAX_COMMENTS` is lower. You can still set it explicitly:

```bash
export ARES_MAX_COMMENTS=10
python3 scripts/eval_comment_sample.py \
  --repo fastapi/fastapi \
  --target-comments 100 \
  --max-inspected-prs 300 \
  --min-human-comments 1 \
  --parallelism 6 \
  --seed 7
```

The sampler prefilters merged PRs to reviewable source changes before cloning, so docs-only, lockfile-only, and other non-code PRs do not consume full review budget.
