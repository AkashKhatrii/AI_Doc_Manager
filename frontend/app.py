import streamlit as st
import requests

st.title("Document AI Assistant")

uploaded_files = st.file_uploader(
    "Upload multiple documents (PDFs/Images)", 
    type=["pdf", "png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

category = st.selectbox("Select Category", ["medical", "academic", "legal", "personal"])

if uploaded_files:
    if st.button("Extract Text"):
        with st.spinner("Processing documents..."):
            # Format files as list of tuples
            files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
            data = {"category": category}

            try:
                response = requests.post("http://localhost:8000/upload/", files=files, data=data)
                response.raise_for_status()

                result = response.json()
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(f"{result['chunks_added']} chunks added from {len(uploaded_files)} documents.")
                    st.write("You can now ask questions.")
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")

            


query = st.text_input("Ask a question about your document:")
if st.button("Get answer"):
    with st.spinner("Thinking..."):
        response = requests.post("http://localhost:8000/ask/", data={"query": query}).json()

        if "answer" in response:
            st.write("**Answer:**", response["answer"])
        else:
            st.error(f"{response.get('error', 'Unknown error occurred.')}")