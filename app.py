
import streamlit as st
import joblib

# Load trained model
model = joblib.load("sied_model.joblib")

# Page configuration
st.set_page_config(
    page_title="Mental Health Signal Analysis",
    page_icon="🧠",
    layout="centered"
)

# Header
st.title("🧠 Mental Health Signal Analysis")

st.write(
    "AI-based analysis of social media text "
    "using machine learning."
)

st.divider()

# User input
st.subheader("Text Analysis")

user_text = st.text_area(
    "Enter Thai text for analysis:",
    placeholder="Type your text here...",
    height=150
)

# Analyze button
if st.button("Analyze Text", type="primary"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")

    else:
        prediction = model.predict([user_text])[0]

        st.subheader("Analysis Result")

        st.success(f"Predicted Label: {prediction}")

        # Display probability if available
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba([user_text])[0]

            st.write("Class probabilities:")

            for label, probability in zip(
                model.classes_,
                probabilities
            ):
                st.write(
                    f"Label {label}: {probability:.2%}"
                )

st.divider()

st.caption(
    "This is an academic research prototype. "
    "It does not provide medical diagnosis or clinical risk assessment."
)
