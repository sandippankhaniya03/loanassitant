import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ===== CONFIG =====
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"   # open-source instruction-tuned model

# ===== LOAD MODELS =====
print("🚀 Loading models...")
embedder = SentenceTransformer(EMBED_MODEL)
generator = pipeline("text-generation", model=LLM_MODEL, device_map="auto", pad_token_id=2)

# ===== LOAD DATASET =====
print("📂 Loading dataset embeddings...")
with open("loan_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
texts = [d["text"] for d in data]

# ===== ENCODE LOCALLY =====
print("⚙️ Generating local embeddings...")
embeddings = embedder.encode(texts, normalize_embeddings=True)
print(f"✅ Loaded {len(embeddings)} text chunks.\n")


# ===== HELPER FUNCTIONS =====
def embed_query(query: str):
    """Convert user question into embedding"""
    return embedder.encode([query], normalize_embeddings=True)[0]


def retrieve_top_chunks(query: str, k: int = 3):
    """Retrieve top-k relevant chunks"""
    query_vec = embed_query(query)
    sims = np.dot(embeddings, query_vec)
    top_idx = np.argsort(sims)[-k:][::-1]
    return [texts[i] for i in top_idx]


def generate_answer(query: str):
    """Generate an answer using RAG logic"""
    context_chunks = retrieve_top_chunks(query)
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant specialized in Bank of Maharashtra loan products.
Use only the provided context to answer accurately.

Context:
{context}

Question: {query}

If the answer is not found in the context, say:
"Sorry, I couldn’t find that in our knowledge base."
    """

    result = generator(prompt, max_new_tokens=250, temperature=0.2)
    return result[0]["generated_text"].split("Question:")[-1].strip()


# ===== MAIN LOOP =====
if __name__ == "__main__":
    print("💬 Loan Assistant RAG System Ready!")
    while True:
        user_q = input("\nAsk about a loan (or 'exit'): ")
        if user_q.lower().strip() == "exit":
            print("👋 Exiting. Goodbye!")
            break
        answer = generate_answer(user_q)
        print(f"\n🤖 {answer}\n")
