import streamlit as st
import requests

st.title("Document AI Assistant")

uploaded_file = st.file_uploader("Upload Document (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    if st.button("Extract Text"):
        with st.spinner("Processing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

            try:
                response = requests.post("http://localhost:8000/upload/", files=files)
                response.raise_for_status()

                data = response.json()
                if "error" in data:
                    st.error(f"Error: {data['error']}")
                else:
                    st.text_area("Document processed! Now you can ask questions.")
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
            


query = st.text_input("Ask a question about your document:")
if st.button("Get answer"):
    with st.spinner("Thinking..."):
        response = requests.post("http://localhost:8000/ask/", data={"query": query})

        if response.status_code == 200:
            st.write("AI Answer: ", response.json()["answer"])
        else:
            st.error("Error getting answer.")