import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="rag_collection"
)

data = collection.get()

print("IDs:\n")
print(data['ids'])

print("\nDocuments:\n")
print(data['documents'])

data = collection.get(
    include=["documents", "embeddings"]
)

print("\nEmbeddings Length:\n")
print(len(data['embeddings']))