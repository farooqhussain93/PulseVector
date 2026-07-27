"""Dataset metadata used by the UI."""

from __future__ import annotations

FEATURE_DESCRIPTIONS = [
    {"name": "age", "description": "Age in years", "type": "Numerical"},
    {"name": "sex", "description": "Recorded sex (0 = female, 1 = male)", "type": "Categorical"},
    {"name": "cp", "description": "Chest-pain category (1-4)", "type": "Categorical"},
    {"name": "trestbps", "description": "Resting blood pressure in mm Hg", "type": "Numerical"},
    {"name": "chol", "description": "Serum cholesterol in mg/dl", "type": "Numerical"},
    {"name": "fbs", "description": "Fasting blood sugar above 120 mg/dl", "type": "Categorical"},
    {"name": "restecg", "description": "Resting electrocardiographic category", "type": "Categorical"},
    {"name": "thalach", "description": "Maximum heart rate achieved", "type": "Numerical"},
    {"name": "exang", "description": "Exercise-induced angina", "type": "Categorical"},
    {"name": "oldpeak", "description": "Exercise-induced ST depression", "type": "Numerical"},
    {"name": "slope", "description": "Slope of peak exercise ST segment", "type": "Categorical"},
    {"name": "ca", "description": "Major vessels colored by fluoroscopy", "type": "Categorical"},
    {"name": "thal", "description": "Thalassemia test category", "type": "Categorical"},
]


FORM_OPTIONS = {
    "sex": [(0, "Female"), (1, "Male")],
    "cp": [
        (1, "Typical angina"),
        (2, "Atypical angina"),
        (3, "Non-anginal pain"),
        (4, "Asymptomatic"),
    ],
    "fbs": [(0, "No"), (1, "Yes")],
    "restecg": [(0, "Normal"), (1, "ST-T abnormality"), (2, "Left ventricular hypertrophy")],
    "exang": [(0, "No"), (1, "Yes")],
    "slope": [(1, "Upsloping"), (2, "Flat"), (3, "Downsloping")],
    "ca": [(0, "0 vessels"), (1, "1 vessel"), (2, "2 vessels"), (3, "3 vessels")],
    "thal": [(3, "Normal"), (6, "Fixed defect"), (7, "Reversible defect")],
}
