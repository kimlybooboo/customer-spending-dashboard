import os

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException

from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html
)

from pydantic import BaseModel, Field

from openai import OpenAI


# ============================================================
# PROJECT FOLDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Load local environment variables
load_dotenv(
    BASE_DIR / ".env"
)


# ============================================================
# LOAD TRAINED MODEL FILES
# ============================================================

model = joblib.load(
    BASE_DIR / "random_forest_model.joblib"
)

feature_columns = joblib.load(
    BASE_DIR / "feature_columns.joblib"
)

label_encoder = joblib.load(
    BASE_DIR / "label_encoder.joblib"
)

outlier_limits = joblib.load(
    BASE_DIR / "outlier_limits.joblib"
)


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Customer Spending Score API",
    description=(
        "Predicts whether a customer belongs to the "
        "Low, Average or High Spending Score category."
    ),
    version="1.0",
    docs_url=None,
    redoc_url=None
)


# Custom Swagger documentation
@app.get("/docs", include_in_schema=False)
def custom_swagger_docs():

    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url=(
            "https://unpkg.com/swagger-ui-dist@5/"
            "swagger-ui-bundle.js"
        ),
        swagger_css_url=(
            "https://unpkg.com/swagger-ui-dist@5/"
            "swagger-ui.css"
        )
    )


# Custom ReDoc documentation
@app.get("/redoc", include_in_schema=False)
def custom_redoc_docs():

    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url=(
            "https://unpkg.com/redoc@2/"
            "bundles/redoc.standalone.js"
        )
    )


# ============================================================
# CUSTOMER INPUT MODEL
# ============================================================

class CustomerInput(BaseModel):

    Gender: Literal[
        "Female",
        "Male"
    ]

    Ever_Married: Literal[
        "No",
        "Yes"
    ]

    Age: int = Field(
        ge=18,
        le=89
    )

    Graduated: Literal[
        "No",
        "Yes"
    ]

    Profession: Literal[
        "Artist",
        "Doctor",
        "Engineer",
        "Entertainment",
        "Executive",
        "Healthcare",
        "Homemaker",
        "Lawyer",
        "Marketing"
    ]

    Work_Experience: int = Field(
        ge=0,
        le=14
    )

    Family_Size: int = Field(
        ge=1,
        le=9
    )

    Var_1: Literal[
        "Cat_1",
        "Cat_2",
        "Cat_3",
        "Cat_4",
        "Cat_5",
        "Cat_6",
        "Cat_7"
    ]


class ExplanationInput(BaseModel):

    customer: CustomerInput

    prediction: str

    probabilities: dict[str, float]

    recommendation: str


# ============================================================
# PREPARE CUSTOMER INPUT
# ============================================================

def prepare_customer_data(
    customer: CustomerInput
):

    customer_data = customer.model_dump()

    # Apply the same outlier limits used during training
    for column in [
        "Age",
        "Work_Experience",
        "Family_Size"
    ]:

        customer_data[column] = min(
            max(
                customer_data[column],
                outlier_limits[column]["lower"]
            ),
            outlier_limits[column]["upper"]
        )

    customer_df = pd.DataFrame([
        customer_data
    ])

    # Convert categorical values into dummy variables
    customer_encoded = pd.get_dummies(
        customer_df,
        drop_first=False,
        dtype=int
    )

    # Match the columns used during model training
    customer_encoded = customer_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return customer_encoded


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def home():

    return {
        "message": (
            "Customer Spending Score API is running."
        )
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.post("/predict")
def predict_spending_score(
    customer: CustomerInput
):

    try:

        customer_encoded = prepare_customer_data(
            customer
        )

        predicted_number = int(
            model.predict(
                customer_encoded
            )[0]
        )

        predicted_label = (
            label_encoder.inverse_transform(
                [predicted_number]
            )[0]
        )

        prediction_probabilities = (
            model.predict_proba(
                customer_encoded
            )[0]
        )

        model_classes = (
            model
            .named_steps["rf"]
            .classes_
            .astype(int)
        )

        class_labels = (
            label_encoder.inverse_transform(
                model_classes
            )
        )

        probabilities = {}

        for label, probability in zip(
            class_labels,
            prediction_probabilities
        ):

            probabilities[str(label)] = round(
                float(probability),
                4
            )


        recommendations = {

            "High": (
                "Consider premium products, loyalty rewards, "
                "exclusive launches and personalised offers."
            ),

            "Average": (
                "Consider product bundles, membership rewards "
                "and personalised product recommendations."
            ),

            "Low": (
                "Consider affordable bundles, introductory "
                "discounts and entry-level products."
            )
        }


        return {
            "predicted_spending_score": str(
                predicted_label
            ),

            "probabilities": probabilities,

            "recommendation": recommendations.get(
                str(predicted_label),
                "Review the customer profile before making decisions."
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# GENERATIVE AI EXPLANATION
# ============================================================

@app.post("/explain")
def generate_explanation(
    information: ExplanationInput
):

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        raise HTTPException(
            status_code=503,
            detail=(
                "The OpenAI API key has not been configured."
            )
        )

    try:

        client = OpenAI(
            api_key=api_key
        )

        prompt = f"""
You are helping a retail manager understand a machine-learning
customer Spending Score prediction.

Customer information:
{information.customer.model_dump()}

Predicted Spending Score:
{information.prediction}

Prediction probabilities:
{information.probabilities}

Current recommendation:
{information.recommendation}

Write a short explanation containing:

1. What the prediction means.
2. Why this customer may belong to the predicted category.
3. Two suitable and ethical marketing suggestions.
4. A limitation explaining that this is only a prediction and
   does not guarantee the customer's future spending.

Do not discriminate against the customer based on gender, age,
marital status or profession.

Do not claim that the model predicts an exact spending amount.
"""

        response = client.responses.create(
            model=os.getenv(
                "OPENAI_MODEL",
                "gpt-5-mini"
            ),
            input=prompt,
            max_output_tokens=250
        )

        return {
            "explanation": response.output_text
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to generate explanation: "
                + str(error)
            )
        )