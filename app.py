import streamlit as st
from fitness_chatbot import build_or_load_vectorstore, generate_answer

st.set_page_config(page_title="Health & Fitness Assistant", layout="wide")

st.markdown("""
    <style>
        /* Center all headings and text */
        .center-text {
            text-align: center;
        }

        /* Chat-like answer box */
        .answer-box {
            border-radius: 5px;
            font-size: 1.1em;
        }

        /* Make textarea wider and centered */
        .stTextArea textarea {
            text-align: left;
            font-size: 1.1em;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='center-text'>AI Fitness Buddy</h1>", unsafe_allow_html=True)
st.markdown("<p class='center-text'>Ask about nutrition, workouts, or guidelines!</p>", unsafe_allow_html=True)
st.divider()

@st.cache_resource
def load_vectorstore():
    return build_or_load_vectorstore(rebuild=False)

vs = build_or_load_vectorstore()

query = st.text_area("Ask a question:", "")

if query:
    with st.spinner("Thinking..."):
        response = generate_answer(vs, query)

    st.markdown("<h3>Answer</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer-box'>{response}</div>", unsafe_allow_html=True)


