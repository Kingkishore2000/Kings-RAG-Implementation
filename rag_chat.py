from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# Load embedding model
embedding_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="rag_collection"
)

while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    # Convert query to embedding
    query_embedding = embedding_model.encode(
        query
    ).tolist()

    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_docs = results['documents'][0]

    context = "\n\n".join(retrieved_docs)

    # Create prompt
    prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY from the provided context.

Context:
{context}

Question:
{query}
"""

    # Send to Ollama
    response = ollama.chat(
        model="gemma:2b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\nAI Response:\n")
    print(response['message']['content'])