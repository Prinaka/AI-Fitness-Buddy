import os
import re
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import argparse
from dotenv import load_dotenv
import json

load_dotenv()

INDEX_DIR = "./index/faiss_health_fitness"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 120
HF_TOKEN = os.getenv('HF_TOKEN')

llm = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    openai_api_key=os.environ["HF_TOKEN"],
    openai_api_base="https://router.huggingface.co/v1"
)

# ---------------------- Load data ----------------

with open(r'./data/documents.json', 'r', encoding='utf-8') as d1:
    raw_docs = json.load(d1)
docs = [Document(page_content=d["content"], metadata=d.get("metadata", {})) for d in raw_docs]

def build_or_load_vectorstore(rebuild= False):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if rebuild or not os.path.isdir(INDEX_DIR):
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)      
        chunks = splitter.split_documents(docs)
        vs = FAISS.from_documents(chunks, embedding=embeddings)
        vs.save_local(INDEX_DIR)
    else:
        vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vs

# ---------------------- RAG Chain --------------------

def qa_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": 6})
    prompt = ChatPromptTemplate.from_template("""
    You are an expert fitness coach and certified nutritionist with over 10 years of experience.
    Instructions:
    1. Analyze the user's question to determine response type needed
    2. Provide complete structured plans with specific details
    3. Include proper progression and safety considerations
    4. Base recommendations on scientific evidence
    Context Information: {context}
    User Question: {input}""")
    chain = create_stuff_documents_chain(llm,prompt)
    retrieval_chain = create_retrieval_chain(retriever,chain)
    return retrieval_chain

def generate_answer(vs, query):
    retrieval_chain = qa_chain(vs)
    res = retrieval_chain.invoke({"input": query})
    answer = res.get("answer") or res.get("output_text", "")
    answer = re.sub(r"\[/?INST\]|\</?s\>", "", answer).strip()
    return answer

# ---------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--ask", type=str, default=None)
    args = parser.parse_args()

    vs = build_or_load_vectorstore(rebuild=args.rebuild)
    
    if args.ask:
        print(generate_answer(vs, args.ask))
        return

    while True:
        q = input("Question: ").strip()
        if q.lower() in {"quit","exit"}:
            break
        if q:
            print(generate_answer(vs,q))


if __name__ == "__main__":
    main()
