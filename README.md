# LLM-Guided Symbolic Regression with Uncertainty Quantification

Code for the paper submission. An LLM proposes candidate equations. Each candidate is fitted
and scored by Bayesian evidence. The best structures are kept in a posterior buffer, and
prediction intervals are calibrated by conformal prediction, scaled by how much the retained
structures disagree.

## Overview

```
 problem description + data
            │
            ▼
 ┌──────────────────────┐   feedback (A1: mechanical, A2/A3: LLM critic)
 │ A. Proposal (LLM)    │◄──────────────────────────────────────────┐
 └──────────┬───────────┘                                           │
            ▼                                                       │
 ┌──────────────────────┐                                           │
 │ B. Fit and retain    │  MAP fit, Laplace covariance, BIC         │
 │                      │  evidence, tempered posterior buffer      │
 └──────────┬───────────┘                                           │
            ├───────────────────────────────────────────────────────┘
            ▼  (after the last round)
 ┌──────────────────────┐
 │ C. Predict and       │  BMA mean + conformal intervals scaled by
 │    calibrate         │  the disagreement between structures
 └──────────────────────┘
```

Ablation rungs:

| Rung | Feedback to the proposer |
|------|--------------------------|
| A0   | none |
| A1   | mechanical diagnostics (weak coefficients, residual patterns, unused inputs, repeated families, stagnation) |
| A2   | A1 + an LLM critic that proposes edits, directives and bans, all grounded in the diagnostics |
| A3   | A2 + label-free calibration evidence for the critic (structure disagreement beyond the training range) |

## Code layout

| File | Content |
|------|---------|
| `run_benchmark.py` | Entry point: runs one rung over LLM-SRBench problems and writes results |
| `llmsr_pipeline.py` | Search loop (propose, fit, retain, calibrate) |
| `hypothesis_generators.py` | LLM proposer (OpenAI API or any OpenAI-compatible server) and a template baseline |
| `grammar.py`, `equation_program.py`, `canonical.py` | Expression grammar with protected operators, compilation, structural signatures |
| `parameter_inference.py` | Multi-start least squares, Laplace covariance, evidence |
| `posterior_buffer.py` | Evidence-weighted buffer with automatic tempering |
| `bma_conformal.py` | Bayesian model averaging, split / weighted / normalized / structural conformal |
| `diagnostics.py` | Mechanical feedback (A1) |
| `critic.py` | LLM critic (A2, A3) and validation of its replies |
| `benchmark/` | LLM-SRBench loader, symbolic and numeric metrics, complexity metrics |
| `devset.py`, `dev_set.json` | Fixed 20-problem dev set (5 per LSR-Synth domain) |
| `reporting.py` | CSV / JSONL writers |

## Installation

Python 3.11 was used.

```bash
pip install -r requirements.txt
```

The benchmark data is downloaded automatically from the Hugging Face Hub
(LLM-SRBench, LSR-Synth subsets) on first use.

## LLM backend

The proposer can be any OpenAI-compatible endpoint. The paper uses Llama-3.1-8B-Instruct served
with vLLM:

```bash
pip install vllm
MODEL=meta-llama/Llama-3.1-8B-Instruct PORT=8000 bash scripts/serve_vllm.sh
```

`run_benchmark.py` reads the server address from `--llm_base_url`
(default `http://127.0.0.1:8000/v1`). For the hosted OpenAI API, set `OPENAI_API_KEY`.

## Running

Quick check without an LLM (template proposer, one problem):

```bash
python run_benchmark.py --rung A0 --limit 1 --iterations 2 --batch 4
```

Main configuration (125 LLM calls per problem, 8 candidates per call, structural intervals
with hybrid weighting):

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct
COMMON="--llm local --llm_model $MODEL --iterations 125 --batch 8 \
        --conformal structural --weighting hybrid --workers 14"

# A1 on the held-out problems of one domain
python run_benchmark.py $COMMON --rung A1 --ablation A1 \
    --subset lsr_synth_bio_pop_growth --heldout

# A0, A2, A3: change --rung (and --ablation)
python run_benchmark.py $COMMON --rung A3 --ablation A3 \
    --subset lsr_synth_bio_pop_growth --heldout

# A2/A3 with GPT-4o as the critic (proposer stays local)
export OPENAI_API_KEY=...
python run_benchmark.py $COMMON --rung A2 --ablation A2-gpt4o \
    --critic_llm openai --critic_model gpt-4o --critic_max_tokens 1500 \
    --subset lsr_synth_bio_pop_growth --heldout
```

The four domains are `lsr_synth_bio_pop_growth`, `lsr_synth_chem_react`, `lsr_synth_matsci`
and `lsr_synth_phys_osc`. Use `--dev` instead of `--subset ... --heldout` to run the dev set.

For A2/A3, `--iterations` is the total LLM-call budget: one call in every `--critic_every` (5)
goes to the critic, so 125 calls = 101 proposer rounds + 24 critic calls. With `--critic_extra`
the critic calls are added on top instead.

Other useful options: `--seed`, `--alpha` (default 0.1), `--buffer_size` (24),
`--restarts` (6), `--resume RUN_DIR`, `--only id1,id2`. See `python run_benchmark.py --help`.

## Output

Each run writes a folder under `outputs/`:

| File | Content |
|------|---------|
| `results.jsonl` | One record per problem: best expression, ID/OOD point metrics, coverage and width, posterior structures, search statistics |
| `results.csv` | Same records, flattened |
| `summary.json`, `summary.csv` | Aggregates per domain and overall |
| `conversations/` | Prompts and LLM replies per round |
| `critic/` | Critic inputs, replies and validation outcomes (A2/A3) |
| `config.json` | Full configuration of the run |

## License

MIT
