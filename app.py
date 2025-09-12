import streamlit as st
from fitness_chatbot import build_or_load_vectorstore, generate_answer

st.set_page_config(page_title="Health & Fitness Assistant")

st.title("AI Fitness Buddy")
st.write("Ask about nutrition, workouts, or guidelines!")
st.divider()

@st.cache_resource
def load_vectorstore():
    return build_or_load_vectorstore(rebuild=False)

vs = build_or_load_vectorstore()

query = st.chat_input("Ask a question...")

if query:
    with st.spinner("Thinking..."):
        response = generate_answer(vs, query)
    chatbot = st.chat_message("assistant")
    chatbot.write(response)
