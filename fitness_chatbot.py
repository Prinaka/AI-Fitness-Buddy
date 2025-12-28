import os
import re
from langchain_community.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
import argparse
from dotenv import load_dotenv
import json


load_dotenv()

INDEX_DIR = "./index/faiss_health_fitness"
HF_TOKEN = os.getenv('HF_TOKEN')

llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=os.environ["GROQ_API"],
    openai_api_base="https://api.groq.com/openai/v1")

# ----------------------Load data----------------

with open(r'./data/documents.json', 'r', encoding='utf-8') as d1:
    raw_docs = json.load(d1)
docs = [Document(page_content=d["content"], metadata=d.get("metadata", {})) for d in raw_docs]

def build_or_load_vectorstore(rebuild= False):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if rebuild or not os.path.isdir(INDEX_DIR):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)      
        chunks = splitter.split_documents(docs)
        vs = FAISS.from_documents(chunks, embedding=embeddings)
        vs.save_local(INDEX_DIR)
    else:
        vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vs

# ---------------------- RAG Chain --------------------
def qa_chain(vs,profile_data=None,memory=None):
    if memory is None:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    name = profile_data.get("name", "N/A")
    age = profile_data.get("age", "N/A")
    gender = profile_data.get("gender", "N/A")
    body_type = profile_data.get("body_type", "N/A")
    height = profile_data.get("height", "N/A")
    weight = profile_data.get("weight", "N/A")
    bmi = profile_data.get("bmi", "N/A")
    goals = profile_data.get("goals", {}) or {}
    diet = profile_data.get("diet", {}) or {}

    primary_goal = goals.get("primary", "N/A")
    target = goals.get("target_weight", "N/A")

    preferences = diet.get("preferences", "N/A")
    allergies = diet.get("allergies", "N/A")
    dislikes = diet.get("dislikes", "N/A")

    prompt = ChatPromptTemplate.from_template("""
    You are an enthusiastic AI Fitness Coach. Your tone should be according to user's age {age} and gender {gender}.
    Regard user with {name}

    User details:
    1. body type: {body_type}
    2. height in cm: {height}
    3. weight in kgs: {weight}
    4. bmi: {bmi}
    5. primary fitness goal: {primary_goal}
    6. target weight in kgs: {target}
    7. dietary preferences: {preferences}
    8. dietary allergies: {allergies}
    9. dietary dislikes: {dislikes}
    10. age: {age}
    11. name: {name}
                                              
    Always use the above user details to personalize your answers.
    If the question refers to something mentioned before, use that memory instead of saying "I don't know."
    Use the following conversation history and retrieved context to answer naturally. Do not start with "As mentioned earlier" etc.

    Instructions:
    1. Analyze the user's question to determine response type needed. Warn user about in case of unsafe/unhealthy requests.
    2. Provide complete structured plans with specific details. Suggest plans only if they will result in healthy outcomes.
    3. Be concise and clear in your responses.
    4. Base recommendations on scientific evidence and best practices.
    5. If user asks for meal plans or diet plans or fitness plans, provide in structured table format.
    6. Always prioritize user safety and well-being in all responses.
                                              
    Chat History:
    {chat_history}

    Retrieved Context:
    {context}

    Question:
    {question}""").partial(
        name=name,
        age=age,
        gender=gender,
        body_type=body_type,
        height=height,
        weight=weight,
        bmi=bmi,
        primary_goal=primary_goal,
        target=target,
        preferences=preferences,
        allergies=allergies,
        dislikes=dislikes)
    
    retrieval_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vs.as_retriever(search_kwargs={"k": 8}), 
                                                            memory=memory, combine_docs_chain_kwargs={"prompt": prompt}, 
                                                            chain_type="stuff", verbose=False)
    return retrieval_chain


def generate_answer(chain, query):
    res = chain.invoke({"question": query})
    answer = res.get("answer") or res.get("output_text", "")
    answer = re.sub(r"\[/?INST\]|\</?s\>", "", answer).strip()
    return answer

# ----------------------CLI-------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--ask", type=str, default=None)
    args = parser.parse_args()

    vs = build_or_load_vectorstore(rebuild=args.rebuild)
    retrieval_chain = qa_chain(vs)
    
    if args.ask:
        print(generate_answer(retrieval_chain, args.ask))
        return

    while True:
        q = input("Question: ").strip()
        if q.lower() in {"quit","exit"}:
            break
        if q:
            print(generate_answer(vs,q))


if __name__ == "__main__":
    main()


