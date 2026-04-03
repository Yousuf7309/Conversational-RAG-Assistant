from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class TextProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process_documents(self, documents):
        if not documents:
            return []
        return self.splitter.split_documents(documents)

# --- Testing Block ---
if __name__ == "__main__":
    
    processor = TextProcessor(chunk_size=100, chunk_overlap=20)
    
    test_doc = Document(page_content="This is a very long sentence used to test if the text processor can successfully split a document into smaller pieces while maintaining a small amount of overlap between them for context.")
    
    chunks = processor.process_documents([test_doc])
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk.page_content}\n")