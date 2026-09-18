# Model Extraction Attack

Implementation of equation-solving attack against a multiclass logistic-regression Iris classifier.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## Run the black-box API

```powershell
py api.py
```

The API listens on `http://127.0.0.1:5000`. The attacker sees only the `/predict` JSON response; model parameters remain in `victim.py`.

## Extract the model

In a second terminal:

```powershell
py extractor.py --queries 12
```

The extractor sends random, linearly independent feature vectors, converts class-probability ratios with `log(p_i / p_r)`, and solves the resulting linear systems by least squares.

## Defensive Counter Measures
In order to activate defensive countermeasures (rounding output probabilities upto 2 decimal places). Use below command:
```powershell
py api.py --decimal 2
```
This reports probability and parameter error for full precision through 2 decimal places. Very low rounded probabilities can become zero, making the logarithm unavailable; the script intentionally reports that condition instead of silently claiming exact recovery.

