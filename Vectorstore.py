from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class VectorStoreManager:
    def __init__(self):
        print("Initializing Embedding Model (This may take a moment)...")
        # UPGRADE: Using a top-tier open-source model for high retrieval accuracy
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5"
        )
        self.vectorstore = None

    def create_vectorstore(self, chunks):
        """Saves chunks into an IN-MEMORY database, avoiding all file locks!"""
        print(f"Creating new in-memory vector store with {len(chunks)} chunks...")
        
        # Notice we removed persist_directory entirely!
        # This keeps the database purely in your Mac's RAM.
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        print("In-Memory Vector store created successfully.")

    def get_retriever(self, num_results=3):
        """Returns the search engine interface for the database."""
        if not self.vectorstore:
            print("Error: Vector store not created yet.")
            return None
        return self.vectorstore.as_retriever(search_kwargs={"k": num_results})

# --- Testing Block ---
if __name__ == "__main__":
    # 1. Initialize the Vector Store Manager
    manager = VectorStoreManager()
    
    # 2. Create some dummy chunks to test the system
    test_chunks = [
        Document(page_content="Generative AI is a type of artificial intelligence technology.", metadata={"source": "test_data"}),
        Document(page_content="Vector databases store data as high-dimensional embeddings.", metadata={"source": "test_data"}),
        Document(page_content="Python is highly recommended for building AI applications.", metadata={"source": "test_data"})
    ]
    
    # 3. Create the database
    manager.create_vectorstore(test_chunks)
    
    # 4. Test the Retriever
    print("\n--- Testing Retrieval ---")
    retriever = manager.get_retriever(num_results=1) 
    
    question = "How is data stored in a vector database?"
    print(f"Query: {question}")
    
    results = retriever.invoke(question)
    
    if results:
        print(f"Best Match Found: {results[0].page_content}")
    else:
        print("No match found.")