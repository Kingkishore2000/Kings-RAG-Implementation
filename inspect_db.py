import chromadb

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

# Get collection
collection = client.get_collection(
    name="rag_collection"
)

# Fetch data
data = collection.get(
    include=["documents", "embeddings", "metadatas"]
)

# Print total records
print(f"\nTotal Records: {len(data['ids'])}\n")

# Print first 3 records
for i in range(min(3, len(data['ids']))):

    print("=" * 80)

    print(f"ID: {data['ids'][i]}")

    print("\nDOCUMENT:\n")
    print(data['documents'][i][:300])

    print("\nEMBEDDING VECTOR SIZE:")
    print(len(data['embeddings'][i]))

    print("\nFIRST 20 EMBEDDING VALUES:")
    print(data['embeddings'][i][:20])

    print("\nMETADATA:")
    print(data['metadatas'][i])