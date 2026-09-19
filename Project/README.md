# Model Extraction Attack on a Black-Box Classifier

> CSE 406 — Computer Security Sessional · Project

A hands-on implementation of the **equation-solving model extraction attack**
(Tramèr et al., *Stealing Machine Learning Models via Prediction APIs*, USENIX
Security 2016) against a multinomial logistic-regression classifier served
behind a REST prediction API — plus an evaluation of **output rounding** as a
defensive countermeasure.

The victim model is never shared with the attacker. Everything the attacker
learns comes from JSON responses to `POST /predict`.

---

## Table of Contents

- [Threat Model](#threat-model)
- [How the Attack Works](#how-the-attack-works)
- [Project Layout](#project-layout)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Start the victim API](#1-start-the-victim-api)
  - [2. Run the extraction attack](#2-run-the-extraction-attack)
  - [3. Evaluate the rounding defense](#3-evaluate-the-rounding-defense)
  - [4. Label-only attack](#4-label-only-attack)
- [Metrics](#metrics)
- [Results](#results)
- [Tests](#tests)
- [Known Issues](#known-issues)
- [References](#references)

---

## Threat Model

| | |
|---|---|
| **Victim** | Scikit-learn pipeline: `StandardScaler` → `LogisticRegression` (multinomial), trained on the Iris dataset (4 features, 3 classes). Lives entirely in `victim.py`. |
| **Exposure** | A Flask API on `http://127.0.0.1:5000`. `POST /predict` returns `{"label": int, "probs": [p0, p1, p2]}`. |
| **Attacker knowledge** | Number of input features, that the response contains class probabilities, and a plausible input domain. **No** weights, **no** training data, **no** gradients. |
| **Attacker goal** | Recover functionally-equivalent model parameters using as few queries as possible. |
| **Defender lever** | Round the returned probabilities to *d* decimal places (`--decimals`). |

## How the Attack Works

A multinomial logistic model assigns

$$
p_i(x) = \frac{\exp(w_i \cdot x + b_i)}{\sum_k \exp(w_k \cdot x + b_k)}
$$

The softmax is invariant to adding a constant to every logit, so the parameters
are only identifiable **up to a reference class**. Picking the last class *r* as
the reference and taking a log-ratio cancels the normalizing denominator and
turns the nonlinear model into a **linear equation in the unknowns**:

$$
\log\left(\frac{p_i(x)}{p_r(x)}\right) = (w_i - w_r) \cdot x + (b_i - b_r)
$$

Every query $$x$$ yields one such equation per class $$i ≠ r$$. With $$d + 1$$
linearly independent queries (here $$4 + 1 = 5$$), the system is exactly
determined; the extractor solves it by **least squares**
(`numpy.linalg.lstsq`) so extra queries simply add robustness.

The recovered $$(w_i − w_r, b_i − b_r)$$ differences reproduce the victim's
probability outputs *exactly* — the un-recoverable shift is unobservable by
construction.

```
attacker                         victim API                 victim model
   │   POST /predict {features}       │                           │
   ├─────────────────────────────────>│──── predict_proba ───────>│
   │   {"label":1,"probs":[...]}      │<──────────────────────────┤
   │<─────────────────────────────────┤
   │
   │  log(p_i / p_r) = (w_i−w_r)·x + (b_i−b_r)   ← one linear equation per query
   │  stack ≥ 5 queries → lstsq → ŵ, b̂          ← stolen model
```

## Project Layout

| File | Role |
|---|---|
| `victim.py` | Trains and holds the secret Iris classifier; exposes `predict()` with optional rounding. |
| `api.py` | Flask black-box server. `POST /predict`, `GET /health`, `--decimals` defense flag. |
| `extractor.py` | The attack: query generation, log-ratio linearization, least-squares solve, and evaluation against ground truth. |
| `experiment.py` | Sweeps the rounding defense (`None, 5, 4, 3, 2` decimals) and reports error at each precision. |
| `label_only.py` | Harder setting: binary-search boundary probing using **labels only**, then an SVD fit of the separating hyperplane. |
| `test_extractor.py` | Pytest check that the solver recovers a synthetic softmax to `1e-10`. |
| `extractor_results.jsonl` | Append-only log of every extraction run (one JSON object per line). |

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Requires Python 3.10+ (the code uses `X | None` type syntax).

## Usage

### 1. Start the victim API

```powershell
py api.py
```

Sanity check it from another shell:

```powershell
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"features\":[5.1,3.5,1.4,0.2]}"
```

### 2. Run the extraction attack

In a second terminal (with the venv active):

```powershell
py extractor.py --queries 12
```

| Flag | Default | Meaning |
|---|---|---|
| `--url` | `http://127.0.0.1:5000/predict` | Oracle endpoint. |
| `--queries` | `12` | Number of oracle queries (needs ≥ 5 to determine the system). |
| `--features` | `4` | Input dimensionality. |
| `--dynamic_queries` | off | Adaptive strategy: sample from a standard normal, clip to the Iris domain, and keep only queries whose probabilities are strictly positive and finite. Useful once the defense starts flattening probabilities to zero. |

Results are **appended** to `extractor_results.jsonl`; the script prints
nothing itself, so inspect that file afterwards.

### 3. Evaluate the rounding defense

Start the API with a rounding precision:

```powershell
py api.py --decimals 2
```

Then sweep precisions and print a JSON report:

```powershell
py experiment.py --queries 30
```

**Why rounding hurts the attack:** the attack needs `log(p_i / p_r)`. Rounding
to *d* places injects an error of up to `0.5 × 10⁻ᵈ` into every probability,
and the logarithm amplifies that error for small probabilities. In the worst
case a genuinely small probability rounds to exactly `0`, the logarithm is
undefined, and the system becomes unsolvable. The code handles this honestly:
`solve_probabilities()` clamps to a `probability_floor` of `0.5 × 10⁻ᵈ`, and
when even that fails the run is recorded with `null` errors rather than
silently claiming exact recovery.

### 4. Label-only attack

The strictly weaker oracle — no probabilities, just the argmax label. The
attacker finds pairs of points on opposite sides of a decision boundary, runs
30 rounds of bisection to land on the boundary, and fits the separating
hyperplane by taking the smallest right-singular vector of the augmented point
matrix.

```powershell
py label_only.py --points 12 --class-a 1 --class-b 2
```


## Metrics

Each run records:

- **`mean_absolute_probability_error`** — mean `|p̂ − p|` over the evaluation
  points. This is the *functional* fidelity of the stolen model, and the number
  that actually matters to an attacker.
- **`parameter_error`** — `‖Ŵ − ΔW‖ + ‖b̂ − Δb‖`, comparing against the victim's
  true weights **folded through the `StandardScaler`** and expressed as
  differences from the reference class.
- **`victim_accuracy`** — agreement between the victim's argmax label and its
  own returned probabilities (a rounding-damage sentinel: it drops below 1.0
  when the defense distorts the output).
- **`attacker_accuracy`** — fraction of points where the stolen model's argmax
  matches the victim's argmax.

## Results

With full-precision probabilities the attack is essentially exact — and it is
exact from the very first sufficient batch of queries. More queries buy
nothing, which is the headline finding:

| Queries | Mean prob. error | Parameter error | Attacker accuracy |
|--:|--:|--:|--:|
| 10 | 2.3e-16 | 1.1e-14 | 1.00 |
| 50 | 3.7e-16 | 1.4e-14 | 1.00 |
| 500 | 8.8e-16 | 1.1e-14 | 1.00 |
| 1000 | 5.5e-16 | 1.5e-14 | 1.00 |

Errors at the `1e-14`–`1e-16` level are floating-point noise: **a model trained
on 150 samples is stolen with ~10 queries, to machine precision.**

The `--dynamic_queries` strategy reaches the same probability fidelity but
shows a much larger raw `parameter_error` (≈22), because clipping samples to
the Iris domain makes the clipped coordinates partially collinear — the solved
parameters drift along a poorly-identified direction while remaining
functionally identical (attacker accuracy stays at 1.00). A useful reminder
that **parameter error and functional equivalence are different things**.

Run `experiment.py` to reproduce the rounding-defense curve: the attack
degrades as precision drops and fails outright once probabilities round to
zero.

## Tests

```powershell
py -m pytest -q
```


*Coursework and defensive research. The victim here is a local toy model — do
not point these scripts at systems you do not own or have permission to test.*
