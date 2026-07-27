<p align="center">
  <img src="static/favicon.svg" alt="PulseVector logo" width="92">
</p>

<h1 align="center">PulseVector</h1>

<p align="center">
  <strong>FastAPI & ML-Powered Heart Disease Classification Engine</strong>
</p>

<p align="center">
  A reproducible binary-classification platform that compares five machine-learning pipelines, selects the winner using training-only cross-validation, independently verifies held-out metrics, and serves validated predictions through FastAPI and Jinja2.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" alt="Python 3.13">
  <img src="https://img.shields.io/badge/FastAPI-0.128-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Tests-71%20Passed-2EA44F" alt="71 tests passed">
  <img src="https://img.shields.io/badge/Coverage-95.98%25-2EA44F" alt="95.98 percent coverage">
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="MIT License">
</p>

> **Educational disclaimer:** PulseVector is an educational machine-learning demonstration and is not a medical diagnostic tool. It must not be used for diagnosis, treatment, clinical decision-making, or emergency guidance.

---

## Live Demo

The Render deployment URL will be added after the production deployment is completed and verified.

**Project Preview:** [Pulse Vector - Preview.pdf](docs/Pulse%20Vector%20-%20Preview.pdf)  
**Technical Documentation:** [PulseVector_Documentation.docx](docs/PulseVector_Documentation.docx)

---

## Screenshots

### Dashboard — Project Overview

![PulseVector Dashboard](screenshots/01-dashboard.png)

### Exploratory Data Analysis — Heart Disease Insights

![PulseVector Exploratory Data Analysis](screenshots/03-eda.png)

### Model Performance — Classification Algorithm Comparison

![PulseVector Model Comparison](screenshots/04-models.png)

### Live Prediction Result — Heart Disease Classification

![PulseVector Prediction Result](screenshots/06-prediction-result.png)

Additional screens are available in the [`screenshots/`](screenshots/) directory:

- Dataset audit
- Prediction form
- About and methodology

---

## Project Overview

PulseVector transforms a supervised binary-classification model into a complete, reproducible machine-learning system. The project covers dataset inspection, leakage-safe preprocessing, five-model comparison, cross-validation-based winner selection, held-out evaluation, artifact verification, FastAPI serving, automated testing, CI configuration, responsive UI, and deployment readiness.

One approved dataset, one target mapping, one stratified train/test split, and the same materialized five-fold `StratifiedKFold` splits are used across all candidate models.

### Selection Policy

1. Highest mean cross-validation F1-score
2. Recall as the first tie-breaker
3. Accuracy as the second tie-breaker

The held-out test set is never used to select the winning model.

### Selected Model

| Item | Result |
|---|---:|
| Winning model | **Logistic Regression** |
| Mean CV F1-score | **0.825** |
| Held-out accuracy | **0.885** |
| Held-out precision | **0.839** |
| Held-out recall | **0.929** |
| Held-out F1-score | **0.881** |
| Held-out specificity | **0.848** |
| Held-out ROC-AUC | **0.966** |

---

## Main Features

- Five classification algorithms in one reproducible project
- Leakage-safe, model-specific scikit-learn pipelines
- Shared stratified cross-validation folds for every candidate
- CV-only winner selection with held-out test-set isolation
- Independently verifiable saved test predictions
- Saved candidate pipelines and `best_model.joblib`
- Accuracy, precision, recall, F1, specificity, ROC-AUC, and confusion matrices
- Interactive Plotly EDA and model-evaluation reports
- FastAPI JSON and HTML prediction endpoints
- Strict Pydantic validation with unknown-field rejection
- Controlled `422`, `503`, and sanitized `500` behavior
- Honest `/health` and end-to-end `/ready` checks
- Responsive Jinja2 interface with accessible navigation
- Screen-reader chart summaries and reduced-motion support
- 71 meaningful automated tests
- 95.98% verified app and ML coverage
- GitHub Actions CI and Render Blueprint configuration

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Language | Python 3.13 |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Jinja2, HTML5, CSS3, vanilla JavaScript |
| Machine Learning | pandas, NumPy, scikit-learn, joblib |
| Visualization | Plotly |
| Testing | Pytest, pytest-cov, HTTPX |
| CI | GitHub Actions |
| Deployment | Render Blueprint |
| Documentation | Markdown, DOCX, PDF |

---

## Model Results

The following table reports held-out test performance for transparency. These values were **not** used to select the winner.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.885** | **0.839** | **0.929** | **0.881** | **0.966** |
| K-Nearest Neighbors | 0.902 | 0.867 | 0.929 | 0.897 | 0.960 |
| Decision Tree | 0.787 | 0.727 | 0.857 | 0.787 | 0.879 |
| Random Forest | 0.902 | 0.844 | 0.964 | 0.900 | 0.950 |
| Support Vector Machine | 0.885 | 0.839 | 0.929 | 0.881 | 0.964 |

Logistic Regression remained the selected model because it achieved the highest mean cross-validation F1-score. Some candidates performed better on this particular 61-row held-out sample, but using test results to change the winner would violate the documented test-set isolation policy.

---

## Dataset & Attribution

| Property | Details |
|---|---|
| Dataset | Heart Disease — Processed Cleveland |
| Original source | UCI Machine Learning Repository |
| Local file | `data/raw/heart_disease_cleveland.csv` |
| Rows | 303 |
| Input features | 13 |
| Original target | `num`, values 0–4 |
| Binary class 0 | Heart disease not detected |
| Binary class 1 | Heart disease present |
| Missing values | 6 total — `ca`: 4, `thal`: 2 |
| Duplicate rows | 0 |
| Dataset license | CC BY 4.0 |

**Target conversion:**

```python
target = (data["num"].astype(int) > 0).astype(int)
```

The MIT License in this repository applies only to the project code. Dataset use remains subject to the UCI source attribution and CC BY 4.0 terms.

---

## Architecture

```text
PulseVector/
├── .github/workflows/tests.yml
├── app/
│   ├── routes/
│   ├── schemas/
│   └── services/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── ml/
├── models/
├── notebooks/
├── reports/
├── screenshots/
├── scripts/
├── static/
├── templates/
├── tests/
├── README.md
├── render.yaml
└── requirements.txt
```

The ML pipeline generates model and report artifacts during CI and deployment. Generated artifacts are intentionally excluded from Git tracking, while `.gitkeep` preserves the required directories.

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/farooqhussain93/PulseVector.git
cd PulseVector
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Generate and verify ML artifacts

```bash
python scripts/run_pipeline.py
python scripts/verify_artifacts.py
```

Expected output includes:

```text
Pipeline complete. Winner: Logistic Regression; held-out F1: 0.881
Artifacts verified: accuracy=0.885, precision=0.839, recall=0.929, f1=0.881
```

### 5. Run the tests

```bash
pytest --cov=app --cov=ml --cov-branch --cov-report=term-missing --cov-fail-under=90
```

Verified build result:

```text
71 passed
Total coverage: 95.98%
```

### 6. Start the application

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

## Main Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard and readiness summary |
| `/dataset` | GET | Dataset audit and feature contract |
| `/eda` | GET | Exploratory data analysis |
| `/models` | GET | Five-model comparison and evaluation |
| `/predict` | GET | Validated HTML prediction form |
| `/prediction/` | POST | HTML prediction submission |
| `/api/predict` | POST | JSON prediction endpoint |
| `/about` | GET | Architecture, methodology, and limitations |
| `/health` | GET | Process liveness |
| `/ready` | GET | End-to-end prediction readiness |
| `/docs` | GET | Interactive OpenAPI documentation |

---

## Health and Readiness

### `/health`

Confirms that the application process is running. It remains available even when prediction artifacts are missing.

### `/ready`

Returns `200` only when PulseVector can make a valid prediction. It checks:

- Model availability and successful loading
- Metrics availability and parsing
- Class mapping
- Exact feature schema
- Winner metadata
- A real end-to-end prediction probe

Unavailable or corrupt required artifacts produce a controlled `503` response rather than fake readiness.

---

## Testing and Quality

- **71/71 tests passed**
- **95.98% combined app and ML coverage**
- **97% coverage for `ml/train.py`**
- Branch coverage enabled
- Coverage threshold enforced at 90%
- Tests use isolated temporary artifacts
- Missing and corrupt models are tested
- Missing and malformed reports are tested
- Winner selection and test-set isolation are tested
- Saved-model reload and prediction parity are tested
- Non-finite and invalid inputs return controlled validation responses

---

## Project Scope

PulseVector demonstrates the engineering lifecycle around a classification model:

- Dataset validation
- Reproducible preprocessing
- Model comparison
- Leakage prevention
- Cross-validation-based model selection
- Held-out evaluation
- Artifact serialization and verification
- Prediction serving
- Failure handling
- Automated testing
- CI configuration
- Deployment readiness
- Responsive portfolio presentation

The project is educational and has not been clinically validated.

---

## Future Improvements

- External validation on an independently approved dataset
- Probability-calibration analysis without test leakage
- Carefully documented model-explainability views
- Drift monitoring for input schema and distributions
- Containerized deployment where required
- Production observability and structured logging
- Real-world validation under an appropriate clinical and regulatory framework

---

## Documentation

- [Technical Documentation](docs/PulseVector_Documentation.docx)
- [Project Preview PDF](docs/Pulse%20Vector%20-%20Preview.pdf)
- [All Project Screenshots](screenshots/)

---

## License

Project code is released under the [MIT License](LICENSE).

The UCI Heart Disease dataset remains subject to its original attribution and CC BY 4.0 terms.
