from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Load both PDFs
documents = []
pdf_files = ["data/sample.pdf", "data/India_States_and_Capitals.pdf" , "data/Virat_Kohli_Short_Notes.pdf"]
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())
    print(f"Loaded {pdf_file}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")

# Embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create Chroma client
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="rag_collection")

# Store embeddings
for i, chunk in enumerate(chunks):
    embedding = embedding_model.encode(chunk.page_content).tolist()

    collection.add(
        ids=[str(i)],
        documents=[chunk.page_content],
        embeddings=[embedding]
    )

print("Embeddings stored successfully")