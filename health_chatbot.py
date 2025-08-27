import os
import pandas as pd
import re
from langchain.chat_models.bedrock import BedrockChat
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import boto3
import botocore
import xml.etree.ElementTree as ET
import argparse


INDEX_DIR = "./index/faiss_health_fitness"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 120

DATA_NUTRITION_DIR = "./data/nutrition"
DATA_EXERCISES_DIR = "./data/exercises"
DATA_GUIDELINES_DIR = "./data/guidelines"
PMC_BUCKET = "pmc-oa-opendata"
PMC_PREFIX = "oa_comm/xml/" 

llm = BedrockChat(
    model_id="meta.llama3-8b-instruct-v1:0",
    region_name="us-east-1",
    model_kwargs={"temperature": 0.7,}
    )

# ---------------------- Load data ----------------

def load_nutrition_df():
    if not os.path.isdir(DATA_NUTRITION_DIR):
        return pd.DataFrame()

    values_file = os.path.join(DATA_NUTRITION_DIR, "ingredient_nutrient_value.csv")
    nutrient_meta_file = os.path.join(DATA_NUTRITION_DIR, "nutrient.csv")
    source_file = "https://fdc.nal.usda.gov/download-datasets"
    df_values = pd.read_csv(values_file).iloc[:,:4]
    df_meta = pd.read_csv(nutrient_meta_file).iloc[:,1:4]
    
    df_values = df_values.merge(
        df_meta[['nutrient_nbr', 'name']],      
        left_on='Nutrient code',      
        right_on='nutrient_nbr',                
        how='left'                    
    ).rename(columns={'name': 'nutrient name'}).drop(columns=['nutrient_nbr'])

    df_wide = df_values.pivot_table(
        index=['ingredient code', 'Ingredient description'],
        columns='nutrient name',
        values='Nutrient value',
        aggfunc='mean'  
    ).reset_index()

    rename_map = {
        "ingredient code": "id",
        "Ingredient description": "name",
        "Energy": "calories_per_100g",
        "Protein": "protein_g_per_100g",
        "Total lipid (fat)": "fat_g_per_100g",
        "Carbohydrate, by difference": "carbs_g_per_100g",
        "Fiber, total dietary": "fiber_g_per_100g"
    }
    existing_cols = {k: v for k, v in rename_map.items() if k in df_wide.columns}
    df_wide.rename(columns=existing_cols, inplace=True)

    df_wide["name_norm"] = df_wide["name"].str.lower().str.strip()
    df_wide["source_file"] = source_file

    cols = ["name", "name_norm", "source_file"]
    for c in ["calories_per_100g", "protein_g_per_100g", "fat_g_per_100g",
              "carbs_g_per_100g", "fiber_g_per_100g"]:
        if c in df_wide.columns:
            cols.append(c)

    return df_wide[cols]


def load_exercises_df():
    frames= []
    source_file = "https://www.kaggle.com/datasets/niharika41298/gym-exercise-data"
    if not os.path.isdir(DATA_EXERCISES_DIR):
        return pd.DataFrame()
    for fn in os.listdir(DATA_EXERCISES_DIR):
        if not fn.lower().endswith(".csv"):
            continue
        path = os.path.join(DATA_EXERCISES_DIR, fn)
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        for col in ["title", "bodypart", "type"]:
            if col not in df.columns:
                df[col] = None
        df["desc"] = df["desc"].astype(str)
        df["source_file"] = source_file
        frames.append(df[["title", "bodypart", "type", "equipment","desc", "source_file"]])
    if not frames:
        return pd.DataFrame(columns=["title", "bodypart", "type", "equipment","desc", "source_file"])
    return pd.concat(frames, ignore_index=True)


def load_guideline_documents():
    docs= []
    if not os.path.isdir(DATA_GUIDELINES_DIR):
        return docs

    for fn in os.listdir(DATA_GUIDELINES_DIR):
        path = os.path.join(DATA_GUIDELINES_DIR, fn)
        if fn.lower().endswith(".pdf"):
            try:
                docs.extend(PyPDFLoader(path).load())
            except Exception:
                pass
        elif fn.lower().endswith((".txt", ".md", ".markdown")):
            try:
                docs.extend(TextLoader(path, encoding="utf-8").load())
            except Exception:
                pass
    return docs



def load_pmc_articles(limit=10):
    s3 = boto3.client("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED))
    docs = []
    count = 0

    resp = s3.list_objects_v2(Bucket=PMC_BUCKET, Prefix=PMC_PREFIX, MaxKeys=limit*2)

    for obj in resp.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".xml"):
            continue

        try:
            body = s3.get_object(Bucket=PMC_BUCKET, Key=key)["Body"].read().decode("utf-8")
            root = ET.fromstring(body)

            title = " ".join(root.findtext(".//article-title", default="").split())
            abstract = " ".join(root.findtext(".//abstract//p", default="").split())
            body_texts = ["".join(p.itertext()) for p in root.findall(".//body//p")]
            body_text = " ".join(body_texts[:50])

            content = f"Title: {title}\n\nAbstract: {abstract}\n\nBody: {body_text}"
            meta = {"type": "pmc", "source": key, "title": title}

            docs.append(Document(page_content=content, metadata=meta))
            count += 1
        except Exception as e:
            print(f"Skipping {key}, error: {e}")
            continue

        if count >= limit:
            break

    print(f"Loaded {len(docs)} PMC articles from S3.")
    return docs

# ---------------------- Build Vector Store ----------------

def build_documents(nutrition_df, exercises_df, guideline_docs, pmc_docs):
    docs= []
    for _, r in nutrition_df.iterrows():
        content = (
            f"Food: {r.get('name_norm','')}. Per 100g -> "
            f"calories: {r.get('calories_per_100g', 'NA')}, protein_g: {r.get('protein_g_per_100g','NA')}, "
            f"fat_g: {r.get('fat_g_per_100g','NA')}, carbs_g: {r.get('carbs_g_per_100g','NA')}, "
            f"fiber_g: {r.get('fiber_g_per_100g','NA')}."
        )
        meta = {"type": "nutrition", "name": r.get("name", ""), "source": r.get("source_file", "")}
        docs.append(Document(page_content=content, metadata=meta))

    for _, r in exercises_df.iterrows():
        content = (
            f"Exercise: {r.get('title','')}. Body part: {r.get('bodypart','')}. "
            f"Instructions: {r.get('desc','')}"
        )
        meta = {"type": "exercise", "name": r.get("title", ""), "source": r.get("source_file", "")}
        docs.append(Document(page_content=content, metadata=meta))

    for d in guideline_docs:
        meta = dict(d.metadata)
        meta.update({"type": "guideline"})
        docs.append(Document(page_content=d.page_content, metadata=meta))

    for d in pmc_docs:
        meta = dict(d.metadata)
        meta.update({"type": "guideline"})
        docs.append(Document(page_content=d.page_content, metadata=meta))
    return docs


def build_or_load_vectorstore(rebuild= False):
    nutrition_df = load_nutrition_df()
    exercises_df = load_exercises_df()
    guidelines_docs = load_guideline_documents()
    pmc_docs = load_pmc_articles(limit=50)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if rebuild or not os.path.isdir(INDEX_DIR):
        print("Building FAISS index ...")
        base_docs = build_documents(nutrition_df, exercises_df, guidelines_docs, pmc_docs)
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)      
        chunks = splitter.split_documents(base_docs)
        vs = FAISS.from_documents(chunks, embedding=embeddings)
        vs.save_local(INDEX_DIR)
        print(f"Saved index to {INDEX_DIR} with {len(chunks)} chunks.")
    else:
        print("Loading existing FAISS index ...")
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
    answer = res['answer']
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
