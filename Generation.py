import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever

load_dotenv()

class RAGGenerator:
    def __init__(self):
        print("Initializing Conversational LLM (Llama 3.1 via Groq)...")
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)


        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"), # Injects previous chat here
            ("human", "{input}"),
        ])

        qa_system_prompt = (
            "You are an expert AI assistant. Use ONLY the following retrieved "
            "context to answer the user's question. If the context does not contain "
            "the answer, say 'I cannot answer this based on the provided documents.'\n\n"
            "Context: {context}"
        )
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"), # Injects previous chat here
            ("human", "{input}")
        ])

    def get_chain(self, retriever):
        """Links the retriever, history reformulator, and answer generator."""
        
        # Step 1: Create the smart retriever that rewrites questions based on history
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, self.contextualize_q_prompt
        )

        # Step 2: Create the standard chain that answers the question
        question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

        # Step 3: Link them together into one seamless Conversational Chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain