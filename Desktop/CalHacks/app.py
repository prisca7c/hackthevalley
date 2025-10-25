import streamlit as st
import json
import os
from backend import analyze_bias

st.set_page_config(page_title="FairCare - Medical Bias Checker", layout="centered")

st.title("🩺 FairCare – Bias Detection in Medical Notes")

# Upload or select sample note
option = st.radio("Choose input method:", ["Upload file", "Select sample", "Manual input"])

if option == "Upload file":
    uploaded_file = st.file_uploader("Upload your .txt file")
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")

elif option == "Select sample":
    files = os.listdir("sample_notes")
    selected = st.selectbox("Choose a sample note", files)
    with open(f"sample_notes/{selected}", "r") as f:
        text = f.read()

else:
    text = st.text_area("Paste or type your note below:")

# Analyze button
if st.button("Analyze for Bias"):
    if text:
        st.write("🔍 Analyzing...")
        result = analyze_bias(text)
        try:
            data = json.loads(result)
        except:
            st.error("Could not parse AI response.")
            st.write(result)
        else:
            st.subheader("⚖️ Bias Check Results")
            st.write("**Fairness Score:**", data.get("score", "N/A"))
            st.write("**Biased Phrases:**")
            st.write(data.get("biased_phrases", []))
            st.write("**Neutral Alternatives:**")
            st.write(data.get("suggestions", []))
    else:
        st.warning("Please enter or upload text first.")
