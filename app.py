from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Keep salary.csv beside this app.py file so local and cloud paths work.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "salary.csv"

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered",
)

@st.cache_resource
def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find salary.csv at: {DATA_PATH}. "
            "Place salary.csv in the same folder as app.py."
        )

    data = pd.read_csv(DATA_PATH)
    required = {"experience_years", "education_level", "job_role", "salary"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "salary.csv is missing required columns: " + ", ".join(sorted(missing))
        )

    data = data[list(required)].copy()
    data["experience_years"] = pd.to_numeric(
        data["experience_years"], errors="coerce"
    )
    data["salary"] = pd.to_numeric(data["salary"], errors="coerce")
    data["education_level"] = data["education_level"].astype("string").str.strip()
    data["job_role"] = data["job_role"].astype("string").str.strip()
    data = data.dropna()
    data = data[
        (data["experience_years"] >= 0)
        & (data["salary"] >= 0)
        & (data["education_level"] != "")
        & (data["job_role"] != "")
    ]

    if len(data) < 2:
        raise ValueError("salary.csv needs at least two valid data rows.")

    X = data[["experience_years", "education_level", "job_role"]]
    y = data["salary"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["education_level", "job_role"],
            )
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)

    education_options = sorted(data["education_level"].unique().tolist())
    role_options = sorted(data["job_role"].unique().tolist())
    max_experience = max(40, int(data["experience_years"].max()) + 10)

    return pipeline, education_options, role_options, max_experience


st.title("💼 Employee Salary Prediction")
st.write("Enter the employee details below to estimate the salary.")

try:
    model, education_options, role_options, max_experience = train_model()
except Exception as exc:
    st.error("The application could not load the model or dataset.")
    st.write("**How to fix it:** Make sure `salary.csv` is in the same folder as `app.py` and has the columns `experience_years`, `education_level`, `job_role`, and `salary`.")
    st.exception(exc)
    st.stop()

with st.form("salary_prediction_form"):
    experience = st.number_input(
        "Experience (years)",
        min_value=0.0,
        max_value=float(max_experience),
        value=2.0,
        step=1.0,
    )
    education = st.selectbox("Education", education_options)
    job_role = st.selectbox("Job Role", role_options)
    submitted = st.form_submit_button("🔮 Predict Salary", use_container_width=True)

if submitted:
    input_data = pd.DataFrame(
        [{
            "experience_years": float(experience),
            "education_level": education,
            "job_role": job_role,
        }]
    )
    prediction = float(model.predict(input_data)[0])
    st.success("Salary prediction completed!")
    st.metric("💰 Predicted Salary", f"₹{prediction:,.0f}")
    st.caption(
        "This is an estimate based on the supplied demo dataset, not a guaranteed salary."
    )
