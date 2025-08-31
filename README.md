# Fitness Chatbot

Fitness Chatbot is an interactive, AI-powered web application that provides personalised answers to questions about nutrition, workouts, and health guidelines using retrieval-augmented generation (RAG) techniques. The assistant leverages advanced language models, a vector store database, and scientific resources to offer accurate, practical fitness coaching and nutrition advice.

**Key Features:**
* Conversational Q&A: Ask freeform questions about fitness, nutrition, guidelines, and workout planning.
* Rich Data Integration: Nutritional values, exercise information, health guidelines, and scientific articles are integrated for reliable responses.
* Structured Fitness Plans: Fitness plan requests are detected and answered in structured, actionable formats.
* Evidence-Based Advice: All responses are grounded in scientific context and best practices.
* Scalable RAG Backend: Combines local CSV, PDF, TXT, and S3-hosted PMC articles for knowledge retrieval.

**Installation:**

1. Clone the repository:
```
gh repo clone Prinaka/Fitness-Chatbot
cd fitness-chatbot
```

2. Create and activate a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Configure AWS credentials (for Bedrock access):
```
aws configure
#Provide AWS Access Key, Secret Key, and default region (us-east-1)
```
5. Prepare Data Folders:
* Place CSVs for nutrition in ./data/nutrition.
  - ingredient_nutrient_value.csv
  - nutrient.csv
* Place exercise CSV files in ./data/exercises.
* Place guidelines PDFs/TXT/MD in ./data/guidelines.
* For PMC articles, ensure AWS S3 connectivity (bucket: pmc-oa-opendata, prefix: oa_comm/xml/).

**Usage:**

Run the Streamlit app:
```
streamlit run app.py
```
* The app launches a modern interface for asking health and fitness questions.
* Ask any question in the text area; responses appear in a styled answer box.
* Fitness plan queries are auto-detected and presented with enhanced formatting.

**Command-Line Interface (CLI):**

Directly use the fitness_chatbot.py for terminal-based Q&A or index rebuilding:
```
python fitness_chatbot.py --ask "How many calories in a banana?"
python fitness_chatbot.py --rebuild
```

**File Overview:**

app.py

* Streamlit UI
* Handles question input, detects plan requests, and displays answers
* Calls backend functions for RAG-based response retrieval

fitness_chatbot.py
* Loads and processes nutrition, exercise, and guideline data
* Downloads and parses PMC articles from S3
* Builds (or reloads) a FAISS vector index for efficient semantic search
* Implements the Q&A chain powered by Llama 3 BedrockChat
* Provides reusable answer generation logic for both CLI and web frontends

**Architecture:**

* Data Loading: Nutrition, exercise, guidelines (local), PMC articles (S3 public bucket).
* Index Building: Documents are chunked and embedded using HuggingFace models, stored in FAISS for fast RAG retrieval.
* LLM Chain: User queries are answered using a retrieval chain with Llama 3 (via LangChain).
* Frontend/CLI: The web interface offers a rich chatbox; CLI enables terminal interaction (also for index rebuilding).

**Customization & Extending:**

* To add new data, place files in the respective directories—no code changes required for core formats.
* The application can be extended with additional health data, exercise types, or guideline documents.
* Adjust model parameters in fitness_chatbot.py:llm for more or less creativity, output length, etc.

**License:**

This project is licensed under the MIT License – see the LICENSE file for details.

**Credits:**

* Food nutrition dataset courtesy USDA: https://fdc.nal.usda.gov/download-datasets
* Exercise dataset courtesy Kaggle: https://www.kaggle.com/datasets/niharika41298/gym-exercise-data
* PMC articles via AWS S3 public bucket: pmc-oa-opendata
