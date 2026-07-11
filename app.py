import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="RetailLens",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "SOURCE(1).csv"


# ============================================================
# BRAND DESIGN
# ============================================================

st.html("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1220px;
}

.brand-logo {
    font-size: 34px;
    font-weight: 900;
    color: #8A3D4D;
    margin-bottom: 0;
}

.brand-tagline {
    color: #7B666B;
    font-size: 15px;
    margin-top: 0;
    margin-bottom: 20px;
}

.hero-section {
    padding: 42px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        #71313F,
        #B85D6D
    );
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 14px 36px
        rgba(113, 49, 63, 0.20);
}

.hero-title {
    font-size: 44px;
    font-weight: 900;
    line-height: 1.12;
    margin-bottom: 14px;
}

.hero-description {
    font-size: 18px;
    line-height: 1.65;
    color: #FFF6F3;
    max-width: 780px;
}

.eyebrow {
    color: #A64B5B;
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 5px;
}

.business-card {
    background-color: #FFFFFF;
    border: 1px solid #E8D8D3;
    border-radius: 18px;
    padding: 24px;
    min-height: 180px;
    margin-bottom: 16px;
    box-shadow: 0 8px 24px
        rgba(80, 45, 53, 0.06);
}

.business-card h3 {
    color: #622C38;
    margin-top: 0;
    margin-bottom: 9px;
}

.business-card p {
    color: #6E5A5F;
    line-height: 1.6;
}

.audience-high {
    border-top: 6px solid #7C3041;
}

.audience-average {
    border-top: 6px solid #CE8A61;
}

.audience-low {
    border-top: 6px solid #D8B4AA;
}

.result-card {
    background: linear-gradient(
        135deg,
        #FFF4EF,
        #F5DDD9
    );
    border: 1px solid #DFBAB4;
    border-radius: 22px;
    padding: 30px;
    margin-top: 20px;
    margin-bottom: 24px;
}

.result-label {
    color: #875B64;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.result-value {
    color: #672B38;
    font-size: 38px;
    font-weight: 900;
    margin-top: 5px;
}

.strategy-box {
    background-color: #FFF9F5;
    border: 1px solid #E8D8D3;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

.strategy-box h3 {
    color: #672B38;
    margin-top: 5px;
}

.step-title {
    color: #5D2A35;
    font-size: 23px;
    font-weight: 800;
    margin-top: 30px;
    margin-bottom: 15px;
}

.step-number {
    display: inline-block;
    background-color: #A64B5B;
    color: white;
    width: 32px;
    height: 32px;
    line-height: 32px;
    text-align: center;
    border-radius: 50%;
    margin-right: 8px;
    font-size: 15px;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E8D8D3;
    padding: 17px;
    border-radius: 15px;
    box-shadow: 0 6px 18px
        rgba(80, 45, 53, 0.05);
}

div[data-testid="stMetricLabel"] {
    color: #725C61 !important;
    font-weight: 650;
}

div[data-testid="stMetricValue"] {
    color: #5C2733 !important;
    font-weight: 850;
}

div.stButton > button {
    border-radius: 999px;
    font-weight: 750;
    min-height: 44px;
}

div.stDownloadButton > button {
    border-radius: 999px;
    font-weight: 750;
    min-height: 44px;
}

</style>
""")


# ============================================================
# FASTAPI ADDRESS
# ============================================================

def get_api_url():

    api_url = os.getenv(
        "API_URL",
        "http://127.0.0.1:8000"
    )

    try:

        if "API_URL" in st.secrets:
            api_url = st.secrets["API_URL"]

    except Exception:
        pass

    return api_url.rstrip("/")


API_URL = get_api_url()


# ============================================================
# LOAD CUSTOMER DATABASE
# ============================================================

@st.cache_data
def load_customer_data():

    data = pd.read_csv(SOURCE_FILE)

    # Remove ID because it is only an identifier
    data = data.drop(
        columns=["ID"],
        errors="ignore"
    )

    # Fill missing numerical values
    data["Work_Experience"] = (
        data["Work_Experience"]
        .fillna(
            data["Work_Experience"].median()
        )
    )

    data["Family_Size"] = (
        data["Family_Size"]
        .fillna(
            data["Family_Size"].median()
        )
    )

    # Fill missing categorical values
    categorical_columns = [
        "Ever_Married",
        "Graduated",
        "Profession",
        "Var_1"
    ]

    for column in categorical_columns:

        data[column] = data[column].fillna(
            data[column].mode()[0]
        )

    return data


df = load_customer_data()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

SCORE_ORDER = [
    "Low",
    "Average",
    "High"
]


def get_top_value(series):

    clean_series = series.dropna()

    if clean_series.empty:
        return "Not available"

    return str(clean_series.mode()[0])


def get_segment_profile(spending_score):

    segment = df[
        df["Spending_Score"] == spending_score
    ].copy()

    age_groups = pd.cut(
        segment["Age"],
        bins=[
            17,
            25,
            35,
            45,
            55,
            65,
            100
        ],
        labels=[
            "18–25",
            "26–35",
            "36–45",
            "46–55",
            "56–65",
            "66+"
        ]
    )

    return {
        "customers": len(segment),

        "percentage": (
            len(segment)
            / len(df)
            * 100
        ),

        "average_age": (
            segment["Age"].mean()
        ),

        "main_age_group": (
            get_top_value(age_groups)
        ),

        "top_profession": (
            get_top_value(
                segment["Profession"]
            )
        ),

        "average_family_size": (
            segment["Family_Size"].mean()
        ),

        "average_work_experience": (
            segment["Work_Experience"].mean()
        ),

        "married_percentage": (
            (
                segment["Ever_Married"] == "Yes"
            ).mean()
            * 100
        ),

        "graduated_percentage": (
            (
                segment["Graduated"] == "Yes"
            ).mean()
            * 100
        )
    }


MARKETING_PLANS = {

    "High": {
        "title": "Loyalty and Premium Audience",

        "objective": (
            "Retain valuable customers and increase "
            "long-term customer value."
        ),

        "offers": (
            "Premium bundles, exclusive product launches, "
            "VIP rewards, priority service and early access."
        ),

        "channels": (
            "Personalised WhatsApp messages, loyalty programmes, "
            "email campaigns and direct recommendations."
        ),

        "message": (
            "Thank customers for their loyalty and offer "
            "exclusive value rather than relying only on discounts."
        ),

        "kpi": (
            "Repeat purchase rate, customer retention and "
            "average order value."
        )
    },

    "Average": {
        "title": "Growth Audience",

        "objective": (
            "Increase purchase frequency and encourage customers "
            "to progress towards the High category."
        ),

        "offers": (
            "Product bundles, spend-and-save promotions, "
            "membership benefits and personalised recommendations."
        ),

        "channels": (
            "Email campaigns, social-media retargeting, "
            "WhatsApp reminders and in-store recommendations."
        ),

        "message": (
            "Show customers how they can receive more value "
            "by purchasing related products together."
        ),

        "kpi": (
            "Conversion rate, repeat purchases and movement "
            "from Average to High."
        )
    },

    "Low": {
        "title": "Nurture and Conversion Audience",

        "objective": (
            "Build trust, reduce barriers to buying and encourage "
            "a first or repeat purchase."
        ),

        "offers": (
            "Starter bundles, affordable products, samples, "
            "introductory discounts and first-purchase rewards."
        ),

        "channels": (
            "Educational social content, broad email campaigns, "
            "remarketing and entry-level promotions."
        ),

        "message": (
            "Focus on affordability, usefulness and clear "
            "product benefits."
        ),

        "kpi": (
            "First purchase, campaign conversion and movement "
            "from Low to Average."
        )
    }
}


def display_marketing_plan(spending_score):

    plan = MARKETING_PLANS[
        spending_score
    ]

    st.html(f"""
    <div class="strategy-box">

        <div class="eyebrow">
            Recommended customer strategy
        </div>

        <h3>
            {plan["title"]}
        </h3>

        <p>
            {plan["objective"]}
        </p>

    </div>
    """)

    column1, column2 = st.columns(2)

    with column1:

        st.markdown(
            "#### Recommended offers"
        )

        st.write(
            plan["offers"]
        )

        st.markdown(
            "#### Recommended channels"
        )

        st.write(
            plan["channels"]
        )

    with column2:

        st.markdown(
            "#### Campaign message"
        )

        st.write(
            plan["message"]
        )

        st.markdown(
            "#### Measure success using"
        )

        st.write(
            plan["kpi"]
        )


def api_is_online():

    try:

        response = requests.get(
            API_URL + "/health",
            timeout=2
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html("""
    <div>

        <div class="brand-logo">
            RetailLens
        </div>

        <div class="brand-tagline">
            Know your customers.<br>
            Market with purpose.
        </div>

    </div>
    """)


page = st.sidebar.radio(
    "Business tools",
    [
        "Home",
        "Customer Insights",
        "Predict New Customer",
        "Marketing Playbook"
    ]
)


st.sidebar.divider()


if api_is_online():

    st.sidebar.success(
        "Prediction service online"
    )

else:

    st.sidebar.error(
        "Prediction service offline"
    )

    st.sidebar.caption(
        "Run: uvicorn api:app --reload"
    )


with st.sidebar.expander(
    "About the prediction model"
):

    st.write(
        "**Model:** Random Forest"
    )

    st.write(
        "**Accuracy:** 81.18%"
    )

    st.write(
        "**F1-score:** 81.65%"
    )

    st.write(
        "**ROC-AUC:** 92.28%"
    )

    st.caption(
        "The model predicts a category, not the exact "
        "amount a customer will spend."
    )


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.html("""
    <div class="hero-section">

        <div class="hero-title">
            Understand your customers.<br>
            Market with more confidence.
        </div>

        <div class="hero-description">
            RetailLens turns an existing retail customer database
            into useful business insights. It helps the owner
            understand current customers, predict a new customer's
            Spending Score and choose a more suitable marketing approach.
        </div>

    </div>
    """)


    st.html(
        '<div class="eyebrow">Business journey</div>'
    )

    st.header(
        "From customer data to marketing action"
    )


    column1, column2, column3 = st.columns(3)


    with column1:

        st.html("""
        <div class="business-card">

            <h3>
                1. Understand
            </h3>

            <p>
                Explore the current customer database and identify
                the size, age, profession and characteristics of
                each Spending Score audience.
            </p>

        </div>
        """)


    with column2:

        st.html("""
        <div class="business-card">

            <h3>
                2. Predict
            </h3>

            <p>
                Enter a new customer's information and predict
                whether they are more likely to belong to the
                Low, Average or High Spending Score category.
            </p>

        </div>
        """)


    with column3:

        st.html("""
        <div class="business-card">

            <h3>
                3. Market
            </h3>

            <p>
                Turn the predicted category into a suitable
                business objective, offer, campaign message,
                communication channel and performance target.
            </p>

        </div>
        """)


    st.divider()


    total_customers = len(df)

    low_customers = (
        df["Spending_Score"] == "Low"
    ).sum()

    high_customers = (
        df["Spending_Score"] == "High"
    ).sum()


    st.html(
        '<div class="eyebrow">Current opportunity</div>'
    )

    st.header(
        "What the customer database is showing"
    )


    snapshot1, snapshot2, snapshot3 = (
        st.columns(3)
    )


    snapshot1.metric(
        "Customer Database",
        f"{total_customers:,}"
    )

    snapshot2.metric(
        "Largest Growth Opportunity",
        f"{low_customers:,} Low"
    )

    snapshot3.metric(
        "Premium Audience",
        f"{high_customers:,} High"
    )


    st.info(
        f"""
        The **Low Spending Score category** is the largest part
        of the database, with **{low_customers:,} customers**.
        The business can treat this as a nurturing and conversion
        opportunity using affordable offers and educational marketing.
        """
    )

    st.success(
        f"""
        The database also contains **{high_customers:,} High
        Spending Score customers**. These customers may be
        prioritised for loyalty programmes, premium bundles and
        exclusive campaigns.
        """
    )


# ============================================================
# CUSTOMER INSIGHTS PAGE
# ============================================================

elif page == "Customer Insights":

    st.html(
        '<div class="eyebrow">Current database</div>'
    )

    st.title(
        "Understand Your Existing Customers"
    )

    st.write(
        "Review the three customer audiences and identify "
        "where the business should retain, grow or nurture."
    )


    high_profile = get_segment_profile(
        "High"
    )

    average_profile = get_segment_profile(
        "Average"
    )

    low_profile = get_segment_profile(
        "Low"
    )


    audience1, audience2, audience3 = (
        st.columns(3)
    )


    with audience1:

        st.html(f"""
        <div class="business-card audience-high">

            <h3>
                High · Loyalty Audience
            </h3>

            <p>
                <b>{high_profile["customers"]:,}</b>
                customers<br>

                {high_profile["percentage"]:.1f}%
                of the database
                <br><br>

                Average age:
                <b>{high_profile["average_age"]:.1f}</b>
                <br>

                Top profession:
                <b>{high_profile["top_profession"]}</b>
            </p>

        </div>
        """)


    with audience2:

        st.html(f"""
        <div class="business-card audience-average">

            <h3>
                Average · Growth Audience
            </h3>

            <p>
                <b>{average_profile["customers"]:,}</b>
                customers<br>

                {average_profile["percentage"]:.1f}%
                of the database
                <br><br>

                Average age:
                <b>{average_profile["average_age"]:.1f}</b>
                <br>

                Top profession:
                <b>{average_profile["top_profession"]}</b>
            </p>

        </div>
        """)


    with audience3:

        st.html(f"""
        <div class="business-card audience-low">

            <h3>
                Low · Nurture Audience
            </h3>

            <p>
                <b>{low_profile["customers"]:,}</b>
                customers<br>

                {low_profile["percentage"]:.1f}%
                of the database
                <br><br>

                Average age:
                <b>{low_profile["average_age"]:.1f}</b>
                <br>

                Top profession:
                <b>{low_profile["top_profession"]}</b>
            </p>

        </div>
        """)


    st.divider()


    selected_segment = st.selectbox(
        "Choose an audience to explore",
        [
            "High",
            "Average",
            "Low"
        ]
    )


    profile = get_segment_profile(
        selected_segment
    )


    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )


    metric1.metric(
        "Main Age Group",
        profile["main_age_group"]
    )

    metric2.metric(
        "Top Profession",
        profile["top_profession"]
    )

    metric3.metric(
        "Married Customers",
        f"{profile['married_percentage']:.1f}%"
    )

    metric4.metric(
        "Average Family Size",
        f"{profile['average_family_size']:.1f}"
    )


    selected_data = df[
        df["Spending_Score"] == selected_segment
    ]


    chart1, chart2 = st.columns(2)


    with chart1:

        st.subheader(
            "Main Professions"
        )

        profession_distribution = (
            selected_data["Profession"]
            .value_counts()
            .head(7)
        )

        st.bar_chart(
            profession_distribution
        )


    with chart2:

        st.subheader(
            "Age Group Distribution"
        )

        age_distribution = pd.cut(
            selected_data["Age"],
            bins=[
                17,
                25,
                35,
                45,
                55,
                65,
                100
            ],
            labels=[
                "18–25",
                "26–35",
                "36–45",
                "46–55",
                "56–65",
                "66+"
            ]
        ).value_counts().sort_index()

        st.bar_chart(
            age_distribution
        )


    display_marketing_plan(
        selected_segment
    )


    with st.expander(
        "Explore customer records"
    ):

        profession_options = sorted(
            selected_data["Profession"]
            .dropna()
            .unique()
        )

        selected_professions = st.multiselect(
            "Filter profession",
            profession_options,
            default=profession_options
        )

        filtered_data = selected_data[
            selected_data["Profession"]
            .isin(selected_professions)
        ]

        st.write(
            f"Showing {len(filtered_data):,} customers"
        )

        st.dataframe(
            filtered_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PREDICT NEW CUSTOMER PAGE
# ============================================================

elif page == "Predict New Customer":

    st.html(
        '<div class="eyebrow">Customer prediction</div>'
    )

    st.title(
        "Predict a New Customer's Spending Score"
    )

    st.write(
        "Enter the customer's profile. The Random Forest "
        "model will predict whether the customer is more "
        "likely to belong to the Low, Average or High category."
    )


    st.html("""
    <div class="step-title">

        <span class="step-number">
            1
        </span>

        Enter customer details

    </div>
    """)


    with st.form(
        "prediction_form"
    ):

        form_column1, form_column2 = (
            st.columns(2)
        )


        with form_column1:

            gender = st.selectbox(
                "Gender",
                sorted(
                    df["Gender"]
                    .dropna()
                    .unique()
                )
            )

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=89,
                value=35
            )

            married = st.selectbox(
                "Marital Status",
                sorted(
                    df["Ever_Married"]
                    .dropna()
                    .unique()
                )
            )

            graduated = st.selectbox(
                "Graduated",
                sorted(
                    df["Graduated"]
                    .dropna()
                    .unique()
                )
            )


        with form_column2:

            profession = st.selectbox(
                "Profession",
                sorted(
                    df["Profession"]
                    .dropna()
                    .unique()
                )
            )

            work_experience = st.number_input(
                "Work Experience",
                min_value=0,
                max_value=14,
                value=1
            )

            family_size = st.number_input(
                "Family Size",
                min_value=1,
                max_value=9,
                value=2
            )

            var_1 = st.selectbox(
                "Customer Category Code",
                sorted(
                    df["Var_1"]
                    .dropna()
                    .unique()
                ),
                help=(
                    "Var_1 is an existing customer category "
                    "included in the original dataset."
                )
            )


        submitted = st.form_submit_button(
            "Predict Spending Score",
            type="primary",
            use_container_width=True
        )


    if submitted:

        customer_data = {
            "Gender": gender,
            "Ever_Married": married,
            "Age": int(age),
            "Graduated": graduated,
            "Profession": profession,
            "Work_Experience": int(
                work_experience
            ),
            "Family_Size": int(
                family_size
            ),
            "Var_1": var_1
        }


        try:

            with st.spinner(
                "Analysing the customer profile..."
            ):

                response = requests.post(
                    API_URL + "/predict",
                    json=customer_data,
                    timeout=60
                )

                response.raise_for_status()

                result = response.json()


            st.session_state[
                "customer_data"
            ] = customer_data

            st.session_state[
                "prediction_result"
            ] = result

            st.session_state.pop(
                "ai_explanation",
                None
            )


        except requests.RequestException as error:

            st.error(
                "The prediction service could not be reached."
            )

            st.write(
                "Make sure FastAPI is running in another terminal:"
            )

            st.code(
                "uvicorn api:app --reload"
            )

            st.code(
                str(error)
            )


    if "prediction_result" in st.session_state:

        result = st.session_state[
            "prediction_result"
        ]

        customer_data = st.session_state[
            "customer_data"
        ]

        predicted_score = result[
            "predicted_spending_score"
        ]

        probabilities = result[
            "probabilities"
        ]

        confidence = (
            max(probabilities.values())
            * 100
        )


        st.html("""
        <div class="step-title">

            <span class="step-number">
                2
            </span>

            Review prediction

        </div>
        """)


        st.html(f"""
        <div class="result-card">

            <div class="result-label">
                Predicted customer audience
            </div>

            <div class="result-value">
                {predicted_score} Spending Score
            </div>

            <p>
                Model confidence:
                <b>{confidence:.1f}%</b>
            </p>

        </div>
        """)


        probability_data = pd.DataFrame({
            "Category": (
                list(
                    probabilities.keys()
                )
            ),

            "Probability": [
                probability * 100
                for probability
                in probabilities.values()
            ]
        })


        probability_data["Category"] = (
            pd.Categorical(
                probability_data["Category"],
                categories=SCORE_ORDER,
                ordered=True
            )
        )


        probability_data = (
            probability_data
            .sort_values("Category")
        )


        chart_column, details_column = (
            st.columns([1.5, 1])
        )


        with chart_column:

            st.subheader(
                "Prediction Confidence"
            )

            st.bar_chart(
                probability_data.set_index(
                    "Category"
                )
            )


        with details_column:

            st.subheader(
                "Customer Profile"
            )

            customer_table = pd.DataFrame(
                customer_data.items(),
                columns=[
                    "Feature",
                    "Customer Value"
                ]
            )

            st.dataframe(
                customer_table,
                use_container_width=True,
                hide_index=True
            )


        st.html("""
        <div class="step-title">

            <span class="step-number">
                3
            </span>

            Choose the marketing approach

        </div>
        """)


        display_marketing_plan(
            predicted_score
        )


        st.warning(
            "The prediction supports business decision-making. "
            "It does not guarantee the customer's future spending "
            "or predict an exact monetary amount."
        )


        if st.button(
            "Generate AI Business Explanation",
            type="primary",
            use_container_width=True
        ):

            explanation_request = {
                "customer": customer_data,

                "prediction": predicted_score,

                "probabilities": probabilities,

                "recommendation": result[
                    "recommendation"
                ]
            }


            try:

                with st.spinner(
                    "Generating the business explanation..."
                ):

                    ai_response = requests.post(
                        API_URL + "/explain",
                        json=explanation_request,
                        timeout=90
                    )

                    ai_response.raise_for_status()

                    st.session_state[
                        "ai_explanation"
                    ] = ai_response.json()[
                        "explanation"
                    ]


            except requests.RequestException as error:

                st.error(
                    "The AI explanation could not be generated."
                )

                st.code(
                    str(error)
                )


        if "ai_explanation" in st.session_state:

            st.subheader(
                "AI Business Explanation"
            )

            st.info(
                st.session_state[
                    "ai_explanation"
                ]
            )


# ============================================================
# MARKETING PLAYBOOK PAGE
# ============================================================

elif page == "Marketing Playbook":

    st.html(
        '<div class="eyebrow">Marketing action plan</div>'
    )

    st.title(
        "Marketing Playbook"
    )

    st.write(
        "Choose a Spending Score audience to see who they are "
        "and how the business can communicate with them."
    )


    selected_score = st.selectbox(
        "Choose customer audience",
        [
            "High",
            "Average",
            "Low"
        ]
    )


    profile = get_segment_profile(
        selected_score
    )

    plan = MARKETING_PLANS[
        selected_score
    ]


    st.subheader(
        plan["title"]
    )


    audience1, audience2, audience3, audience4 = (
        st.columns(4)
    )


    audience1.metric(
        "Audience Size",
        f"{profile['customers']:,}"
    )

    audience2.metric(
        "Database Share",
        f"{profile['percentage']:.1f}%"
    )

    audience3.metric(
        "Average Age",
        f"{profile['average_age']:.1f}"
    )

    audience4.metric(
        "Top Profession",
        profile["top_profession"]
    )


    display_marketing_plan(
        selected_score
    )


    st.divider()


    st.subheader(
        "Ready-to-Use Campaign Brief"
    )


    campaign_brief = f"""
RETAILLENS CUSTOMER CAMPAIGN BRIEF

TARGET AUDIENCE
{selected_score} Spending Score customers

AUDIENCE SIZE
{profile['customers']:,} customers
({profile['percentage']:.1f}% of the current database)

CUSTOMER PROFILE
Average age: {profile['average_age']:.1f}
Main age group: {profile['main_age_group']}
Top profession: {profile['top_profession']}
Married customers: {profile['married_percentage']:.1f}%
Average family size: {profile['average_family_size']:.1f}

CAMPAIGN OBJECTIVE
{plan['objective']}

RECOMMENDED OFFERS
{plan['offers']}

RECOMMENDED CHANNELS
{plan['channels']}

CAMPAIGN MESSAGE
{plan['message']}

MAIN KPI
{plan['kpi']}

LIMITATION
This audience profile is based on patterns in the existing
customer database. It should guide marketing decisions but
does not guarantee future customer behaviour.
"""


    st.text_area(
        "Campaign brief",
        campaign_brief,
        height=390
    )


    st.download_button(
        "Download Campaign Brief",
        data=campaign_brief,
        file_name=(
            selected_score.lower()
            + "_customer_campaign_brief.txt"
        ),
        mime="text/plain",
        type="primary",
        use_container_width=True
    )