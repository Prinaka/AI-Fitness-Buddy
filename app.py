import streamlit as st
import os
from dotenv import load_dotenv
from fitness_chatbot import build_or_load_vectorstore, qa_chain, generate_answer
from langchain_classic.memory import ConversationBufferMemory
from supabase import create_client, ClientOptions
from datetime import datetime as date

load_dotenv()

url = os.environ["SUPABASE_URL"]
key = os.environ["ANON_KEY"]  # only anon key used in client

supabase_anon = create_client(url, key)

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

st.set_page_config(page_title="AI Fitness Buddy", layout="centered")

#------------Authentication------------------

if "user" not in st.session_state:
    st.session_state.user = None

auth_box = st.empty()

if st.session_state.user is None:

    with auth_box.container():
        st.title("Welcome to AI Fitness Buddy")
        if "signup_mode" not in st.session_state:
            st.session_state.signup_mode = False
    
        if not st.session_state.signup_mode:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", key="login_password", type="password")
            login = st.button("Login")
            st.caption("Don't have an account?")
            signup = st.button("Sign up")

            if signup:
                st.session_state.signup_mode = "Signup"
                st.rerun()
            
            if login:
                try:
                    session = supabase_anon.auth.sign_in_with_password({"email": email,"password": password})
                    st.session_state.user = session.user
                    st.session_state.access_token = session.session.access_token
                    st.success(f"Welcome, {st.session_state.user.email}!")
                    auth_box.empty()
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")


        if st.session_state.signup_mode == "Signup":
            placeholder = st.empty()
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", key="signup_password",type="password")
            name = st.text_input("Name")
            dob = st.date_input("Date of Birth", value=date(2004, 10, 18))
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            st.image("./data/body_types.png")
            body_type = st.selectbox("Body Type", ["Ectomorph", "Mesomorph", "Endomorph"])
            height = st.number_input("Height (cm)", min_value=120, max_value=200)
            weight = st.number_input("Weight (kg)", min_value=45.0, max_value=150.0, step=0.1)
            bmi = weight / ((height / 100) ** 2)
            min_ideal_weight = round(18.5 * ((height / 100) ** 2), 1)
            max_ideal_weight = round(24.9 * ((height / 100) ** 2), 1)
            goals = {
                "primary": st.multiselect("Primary Goal", ["Weight Loss", "Muscle Gain", "Flexibility", "General Fitness"]),
                "target_weight": st.number_input("Target Weight (kg)", min_value=min_ideal_weight, max_value=max_ideal_weight, step=0.1)
            }
            diet = {
                "preferences": st.selectbox("Dietary Preferences", ["Vegetarian", "Non-Vegetarian", "Vegan", "Keto"]),
                "allergies": st.text_area("Allergies").split(","),
                "dislikes": st.text_area("Dislikes").split(",")
            }

            if st.button("Create Account"):
                try:
                    res = supabase_anon.auth.sign_up({"email": email, "password": password})
                    user_id = res.user.id if res.user else res.session.user.id
                    if user_id:
                        profile_data = {
                            "user_id": user_id,
                            "email": email,
                            "name": name,
                            "dob": dob.isoformat(),
                            "gender": gender,
                            "body_type": body_type,
                            "height": height,
                            "weight": weight,
                            "bmi": round(bmi, 2),
                            "goals": dict(goals),
                            "diet": dict(diet)
                        }
                        supabase_anon.table("USERS").insert(profile_data).execute()
                        st.success("✅ Account created! Please log in.")
                except Exception as e:
                    st.error(f"Error: {e}")


if st.session_state.user:
    st.title("AI Fitness Buddy")
    st.write("Ask about nutrition, workouts, or guidelines!")
    supabase = create_client(url, key, options=ClientOptions(headers={"Authorization": f"Bearer {st.session_state.access_token}"}))
    st.session_state.profile_data = supabase.table("USERS").select("*").eq("user_id", st.session_state.user.id).execute().data[0]
    age = int((date.today() - date.fromisoformat(st.session_state.profile_data["dob"])).days // 365)
    st.session_state.profile_data["age"] = age
    st.divider()

#-----------Chat Interface-----------------

def new_chat():
    st.session_state.chat_counter += 1
    chat_id = f"chat_{st.session_state.chat_counter}"
    st.session_state.chats[chat_id] = {"title": "Default Chat", 
                                       "history": [], 
                                       "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True)}
    st.session_state.current_chat = chat_id


if "chats" not in st.session_state:
    st.session_state.chats = {}  
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 0


if st.session_state.user:
    if st.sidebar.button(label="➕ New Chat", type="secondary", use_container_width=True):
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
    new_chat()

chat_data = st.session_state.chats[st.session_state.current_chat]
chat_history = chat_data["history"]
memory = chat_data["memory"]
profile_data = st.session_state.profile_data if "profile_data" in st.session_state else {}

#-----------Answer Generation-----------------

chain = qa_chain(vs,profile_data,memory)

for msg in chat_history:
    st.chat_message(msg["role"]).write(msg["content"])
    
if st.session_state.user:
    query = st.chat_input("Ask a question...", key=f"input_{st.session_state.current_chat}")

    if query:
        if chat_data["title"] in ["Default Chat"]:
            chat_data["title"] = query[:20] + ("..." if len(query) > 20 else "")

        st.chat_message("human").write(query)
        chat_history.append({"role": "human", "content": query})

        with st.spinner("Thinking..."):
            response = generate_answer(chain, query)

        st.chat_message("assistant").write(response)
        chat_history.append({"role": "assistant", "content": response})
        chat_data["memory"].save_context({"input": query}, {"output": response})
