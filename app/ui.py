import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ESG Analyzer", layout="wide")
st.title("ESG Sustainability Analyzer")
st.write("Upload or paste an ESG report to get a score and SDG alignment.")

text = st.text_area("Paste report text here", height=250, max_chars=50000)

if st.button("Analyze", type="primary") and text:
    with st.spinner("Analyzing..."):
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"text": text},
                timeout=60
            )
            result = resp.json()

            col1, col2 = st.columns(2)

            with col1:
                st.metric("ESG Score", f"{result['esg_score']:.1f}")

            with col2:
                st.subheader("SDGs Detected")
                if result["sdg_labels"]:
                    for sdg in result["sdg_labels"]:
                        st.success(sdg)
                else:
                    st.info("No SDGs detected")

            st.subheader("SDG Probabilities")
            for sdg, prob in result["sdg_probabilities"].items():
                st.progress(prob, text=f"{sdg}: {prob:.1%}")

        except Exception as e:
            st.error(f"Error: {e}")