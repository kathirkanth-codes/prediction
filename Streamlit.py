import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000"

st.title("💰 Bank Spending Prediction")

st.write("Upload your bank statement PDF to predict next month spending.")

user_id = st.text_input("User ID")

uploaded_file = st.file_uploader("Upload Bank Statement PDF", type=["pdf"])

if st.button("Upload and Predict"):

    if uploaded_file is None or user_id == "":
        st.error("Please upload a file and enter user id")
    else:

        # -------------------------
        # Upload PDF
        # -------------------------
        files = {"file": uploaded_file}
        data = {"user_id": user_id}

        upload_response = requests.post(
            f"{API_URL}/upload",
            files=files,
            data=data
        )

        if upload_response.status_code != 200:
            st.error("Upload failed")
        else:
            st.success("PDF uploaded successfully")

            # -------------------------
            # Call prediction API
            # -------------------------
            pred_response = requests.get(
                f"{API_URL}/predict/{user_id}"
            )

            if pred_response.status_code != 200:
                st.error("Prediction failed")
            else:
                result = pred_response.json()

                st.subheader("Prediction Result")

                st.write("Category:", result["category"])
                st.write(
                    "Predicted Next Month Spending:",
                    round(result["predicted_next_month_spending"], 2)
                )
