import streamlit as st
from fitness_chatbot import build_or_load_vectorstore, generate_answer

st.markdown(
    """
    <style>
    div.stButton > button {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.set_page_config(page_title="Health & Fitness Assistant", layout="centered")

st.title("AI Fitness Buddy")
st.write("Ask about nutrition, workouts, or guidelines!")
st.divider()

if "chats" not in st.session_state:
    st.session_state.chats = {}  
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 0
   
def new_chat():
    st.session_state.chat_counter += 1
    chat_id = f"chat_{st.session_state.chat_counter}"
    st.session_state.chats[chat_id] = {"title": "Untitled Chat", "history": []}
    st.session_state.current_chat = chat_id

if st.sidebar.button(label="➕ New Chat", type="secondary", width="stretch"):
    new_chat()

st.sidebar.subheader("Chat History")
for chat_id, chat_data in st.session_state.chats.items():
    label = chat_data["title"]
    if st.sidebar.button(label, key=f"btn_{chat_id}", width="stretch"):
        st.session_state.current_chat = chat_id

@st.cache_resource
def load_vectorstore():
    return build_or_load_vectorstore(rebuild=False)

vs = load_vectorstore()

if st.session_state.current_chat is None:
    st.session_state.chat_counter += 1
    default_chat_id = f"chat_{st.session_state.chat_counter}"
    st.session_state.chats[default_chat_id] = {"title": "Default Chat", "history": []}
    st.session_state.current_chat = default_chat_id
    
if st.session_state.current_chat:
    chat_data = st.session_state.chats[st.session_state.current_chat]
    chat_history = chat_data["history"]

    for msg in chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    query = st.chat_input("Ask a question...", key=f"input_{st.session_state.current_chat}")

    if query:
        if chat_data["title"] == "Untitled Chat":
            chat_data["title"] = query[:20] + ("..." if len(query) > 20 else "")

        st.chat_message("human").write(query)
        chat_history.append({"role": "human", "content": query})

        with st.spinner("Thinking..."):
            response = generate_answer(vs, query)

        st.chat_message("assistant").write(response)
        chat_history.append({"role": "assistant", "content": response})




