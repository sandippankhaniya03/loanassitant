
# Loan Assistant — RAG-based Chatbot for Bank Loan Queries


### 📘 Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers queries related to Bank of Maharashtra loan products.
It combines web-scraped data, embeddings, and open-source LLMs to provide accurate, contextual answers from the bank’s official information.


## ⚙️ Project Setup

1. Clone the Repository

```bash
  git clone https://github.com/sandippankhaniya03/loanassitant.git
```

Go to the project directory

```bash
  cd loanassitant
```

2. Create and Activate Virtual Environment (Windows)

```bash
  python -m venv venv
  venv\Scripts\activate
```

3. Install Dependencies

```bash
  pip install -r requirements.txt
```

4. Run the RAG Chatbot
```bash
  python loan_rag_pipeline.py

```

Then type your questions in the console, for example:
```bash
  Ask about a loan (or 'exit'): What is the processing fee for a home loan?


```

| Purpose             | Library                                               | Reason                                              |
| ------------------- | ----------------------------------------------------- | --------------------------------------------------- |
| **Data Scraping**   | `requests`, `BeautifulSoup`                           | Easy parsing of bank loan pages                     |
| **Data Processing** | `pandas`, `re`                                        | Clean and structure text data                       |
| **Embeddings**      | `sentence-transformers` (`all-MiniLM-L6-v2`)          | Lightweight, fast, and free embedding model         |
| **Vector Search**   | `numpy`                                               | Efficient similarity computations                   |
| **LLM Generation**  | `transformers` (`mistralai/Mistral-7B-Instruct-v0.2`) | Open-source, high-quality instruction-following LLM |

## 🧠 Data Strategy

+ After scraping loan-related FAQs and descriptions, the text was:

+ Cleaned and normalized (removed HTML tags, special characters).

+ Chunked logically — each FAQ, paragraph, or key section became one document chunk.

+ This ensures semantic completeness (each chunk answers one question or concept).

+ Each chunk was embedded using all-MiniLM-L6-v2 and stored in loan_embeddings.pkl

## 🤖 Model Selection

| Component            | Model                                                                                           | Reason                                               |
| -------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **Embedding Model**  | `all-MiniLM-L6-v2`                                                                              | Free, efficient, high cosine similarity accuracy     |
| **LLM Model**        | `mistralai/Mistral-7B-Instruct-v0.2`                                                            | Good balance of reasoning and performance on CPU/GPU |
| **Why Open Source?** | To make the project **completely offline and cost-free** without Gemini or OpenAI dependencies. |                                                      |

## 🧰 AI Tools Used

* SentenceTransformer — to convert text into vector embeddings.

- Hugging Face Transformers — to use an open-source LLM for generating human-like answers.

* RAG Pipeline Design — combines both embeddings and text generation seamlessly.

| Challenge                            | Solution                                        |
| ------------------------------------ | ----------------------------------------------- |
| Dynamic web pages during scraping    | Using You tube Transcipt get infomation about data |
| Mixed text formats (HTML, JSON, PDF) | Standardized into plain text before chunking    |
| Large number of similar FAQs         | Deduplicated embeddings to reduce vector size   |
| LLM response hallucination           | Restricted context strictly to retrieved chunks |

## 📁 Project Structure


```text
loanassitant/
│
├── loan_rag_pipeline.py      # Main RAG chatbot script
├── loan_embeddings.pkl       # Precomputed embeddings
├── dataset/
│   ├── faq.txt
│   ├── faq.csv
│   └── faq.json
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
