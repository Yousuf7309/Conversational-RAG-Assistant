import streamlit as st
import os
import tempfile # <--- New import to handle the PDF memory issue!
from dotenv import load_dotenv

# Import our custom Lego blocks
from Data_Ingestion import DataIngestor
from Processing import TextProcessor
from Vectorstore import VectorStoreManager
from Generation import RAGGenerator

# Load environment variables
load_dotenv()

# ==========================================
# UI Setup & Session State
# ==========================================
st.set_page_config(page_title="Pro RAG Assistant", layout="wide")
st.title("Conversational RAG Assistant")
st.markdown("Powered by **Llama 3.1 (Groq)** and **BGE-Large Embeddings**.")
st.markdown("**Instructions (How to use) -**")
st.markdown("""
1. Go to the sidebar on the left and upload your PDFs or paste website URLs.
2. Click the **Process Documents** button and wait for the success message.
3. Ask follow-up questions in the chat box below!
---
""")

# Initialize session state variables
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================
# Sidebar: Data Upload
# ==========================================
with st.sidebar:
    st.header("1. Upload Knowledge Base")
    
    urls_input = st.text_area("Enter Website URLs (one per line):", height=150)
    st.markdown("**OR / AND**")
    uploaded_files = st.file_uploader("Upload PDF Documents:", type=["pdf"], accept_multiple_files=True)
    
    process_btn = st.button("Process Documents", type="primary")

    if process_btn:
        url_list = [url.strip() for url in urls_input.split('\n') if url.strip()]
        
        if not url_list and not uploaded_files:
            st.warning("Please provide at least one URL or PDF!")
        else:
            with st.spinner("Building Vector Database... (This may take a minute)"):
                try:
                    ingestor = DataIngestor()
                    
                    if url_list:
                        st.info(f"Loading {len(url_list)} URLs...")
                        ingestor.ingest_urls(url_list)
                        
                    if uploaded_files:
                        st.info(f"Loading {len(uploaded_files)} PDFs...")
                        tmp_file_paths = []
                        for uploaded_file in uploaded_files:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_paths.append(tmp_file.name)
                        
                        ingestor.ingest_pdfs(tmp_file_paths)
                        
                        for path in tmp_file_paths:
                            os.remove(path)

                    docs = ingestor.get_documents()
                    processor = TextProcessor(chunk_size=1000, chunk_overlap=200)
                    chunks = processor.process_documents(docs)
                    
                    v_store = VectorStoreManager()
                    v_store.create_vectorstore(chunks)
                    
                    st.session_state.retriever = v_store.get_retriever()
                    st.success(f"Successfully processed {len(chunks)} text chunks!")
                
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                    
    # --- The Reset Button ---
    st.markdown("---") 
    if st.button("🗑️ Clear All & Start Fresh", use_container_width=True):
        st.session_state.retriever = None
        st.session_state.chat_history = []
        st.rerun()

# ==========================================
# Main Chat Interface
# ==========================================
st.header("2. Ask Questions")

# 1. Draw the existing chat history on the screen
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. Wait for the user to ask a new question
user_question = st.chat_input("Ask a question about the documents...")

if user_question:
    if not st.session_state.retriever:
        st.warning("Please upload and process documents in the sidebar first!")
    else:
        # Display the user's question immediately
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Save the user's question to the Streamlit history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # 3. Generate the AI Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    generator = RAGGenerator()
                    chain = generator.get_chain(st.session_state.retriever)
                    
                    langchain_history = []
                    for msg in st.session_state.chat_history:
                        if msg["content"] == user_question:
                            continue
                        if msg["role"] == "user":
                            langchain_history.append(("human", msg["content"]))
                        else:
                            langchain_history.append(("ai", msg["content"]))
                    
                    # Run the chain
                    response = chain.invoke({
                        "input": user_question,
                        "chat_history": langchain_history
                    })
                    
                    # 1. Extract the answer AND the hidden source documents
                    answer = response["answer"]
                    source_documents = response["context"] # <-- THIS IS THE SECRET SAUCE!
                    
                    # 2. Print the main answer to the screen
                    st.markdown(answer)
                    
                    # 3. Build the NotebookLM-style Source Viewer
                    with st.expander("🔍 View Sources & Referenced Text"):
                        st.markdown("The AI used the following chunks of text to generate this answer:")
                        
                        # Loop through the 3 chunks it retrieved
                        for i, doc in enumerate(source_documents):
                            # Extract the hidden metadata
                            source = doc.metadata.get("source", "Unknown Document")
                            page = doc.metadata.get("page", "N/A") # PDFs have pages, URLs don't
                            
                            # Clean up the file path so it looks nice on screen
                            clean_source = os.path.basename(source) if "http" not in source else source
                            
                            st.markdown(f"**Source {i+1}:** `{clean_source}` (Page: {page})")
                            # Show the exact paragraph it read
                            st.info(f'"{doc.page_content}"')
                            
                    # 4. Save to chat history (We save a clean version so the UI doesn't get cluttered)
                    history_note = f"\n\n*(Used {len(source_documents)} sources to answer)*"
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": answer + history_note
                    })
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")