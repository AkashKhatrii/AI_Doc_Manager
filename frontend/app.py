import streamlit as st
import requests

st.title("Document AI Assistant")

uploaded_file = st.file_uploader("Upload Document (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    if st.button("Extract Text"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

        response = requests.post("http://localhost:8000/extract-text/", files=files)

        if response.status_code == 200:
            data = response.json()
            st.text_area("Extracted text", data["text"], height=400)
        else:
            st.error("Error extracting text.")