import streamlit as st
from backend import get_response

st.title("🧠 RAG Chatbot with LangChain")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        response = get_response(query)
        st.write("Response: ", response)
    else:
        st.write("Please enter a question.")
