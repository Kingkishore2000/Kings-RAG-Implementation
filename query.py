from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(name="rag_collection")

query = input("Ask something: ")

query_embedding = embedding_model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\nRetrieved Results:\n")

for doc in results['documents'][0]:
    print(doc)
    print("\n-----------------\n")