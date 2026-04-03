import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

load_dotenv()

class DataIngestor:
    def __init__(self):
        self.all_documents = []

    def ingest_urls(self, urls):
        loader = WebBaseLoader(urls)
        documents = loader.load()
        self.all_documents.extend(documents)

    def ingest_pdfs(self, file_paths):
        for file_path in file_paths:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            self.all_documents.extend(documents)

    def get_documents(self):
        return self.all_documents

# --- Testing Block ---
if __name__ == '__main__':
    ingestor = DataIngestor()
    # Try a Google-hosted URL which is very stable
    test_urls = ['https://www.google.com'] 
    ingestor.ingest_urls(test_urls)
    
    documents = ingestor.get_documents()
    print(f"SUCCESS: Loaded {len(documents)} documents!")