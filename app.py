import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("shopping_model.pkl")

st.title("Online Shopping Intention Analysis")

st.write("Predict whether a customer will purchase a product.")

page_values = st.number_input(
    "Page Values",
    min_value=0.0,
    value=5.0
)

exit_rates = st.number_input(
    "Exit Rates",
    min_value=0.0,
    value=0.05
)

bounce_rates = st.number_input(
    "Bounce Rates",
    min_value=0.0,
    value=0.02
)

if st.button("Predict"):

    sample = pd.DataFrame(
        [[
            0,
            0,
            0,
            0,
            1,
            10,
            bounce_rates,
            exit_rates,
            page_values,
            0,
            1,
            1,
            1,
            1,
            2,
            1,
            0
        ]]
    )

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.success("Customer is likely to Purchase ✅")
    else:
        st.error("Customer is unlikely to Purchase ❌")