# Conversational RAG Assistant

## Overview
An enterprise-grade Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDFs and website URLs, and chat with their documents in real-time. 

Unlike basic RAG pipelines, this application features Conversational Memory and Query Reformulation, allowing it to understand ambiguous follow-up questions (e.g., "What does it do?") by intelligently analyzing the chat history before querying the vector database.

## Key Engineering Features
* History-Aware Retrieval: Uses a secondary LLM step to rewrite user follow-up questions into standalone queries, solving the "Amnesiac AI" problem.
* In-Memory Vector Database: Uses an ephemeral, lock-free ChromaDB implementation in RAM, completely eliminating hard-drive file lock errors and ensuring seamless multi-document uploads.
* Verifiable Citations: Automatically extracts hidden metadata to provide exact document names, page numbers, and quoted text chunks to prevent hallucinations.
* Multi-Modal Ingestion: Capable of scraping active URLs and parsing complex PDF documents simultaneously.

## Architecture & Data Flow
1. Ingestion (Data_Ingestion.py): Extracts raw text from lists of URLs and user-uploaded PDFs using LangChain document loaders.
2. Processing (Processing.py): Splits massive documents into 1,000-character chunks with a 200-character overlap to preserve semantic context.
3. Storage (Vectorstore.py): Converts text chunks into mathematical vectors using the open-source BAAI/bge-large-en-v1.5 embedding model and stores them in an In-Memory Chroma database.
4. Conversational Generation (Generation.py): 
   * Intercepts user questions and reformulates them using chat history.
   * Retrieves the top 3 most relevant text chunks.
   * Generates a final, accurate response using Llama 3.1 8B (via Groq API).
5. Front-End (app.py): A clean, stateful Streamlit interface that manages conversation history and temporary file cleanup.

## Tech Stack
* LLM Engine: Llama 3.1 8B (via Groq)
* Embeddings: HuggingFace BGE-Large
* Orchestration: LangChain
* Vector Database: ChromaDB
* Front-End: Streamlit
* Package Management: uv

## How to Run Locally

1. Clone the repository
git clone https://github.com/Yousuf7309/Conversational-RAG-Assistant.git
cd Conversational-RAG-Assistant

2. Set up your environment
Make sure you have uv installed, then sync the dependencies:
uv sync

3. Add your API Keys
Create a .env file in the root directory and add your Groq API key:
GROQ_API_KEY="gsk_your_api_key_here"
USER_AGENT="MyRAGProject/1.0"

4. Launch the App
uv run streamlit run app.py