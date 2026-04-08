import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class RAGPipeline:
    def __init__(self):
        # Initialize embeddings model (runs locally)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.qa_chain = None
        
        # Initialize LLM. Will require GROQ_API_KEY in environment
        # We use a fast and capable model from Groq
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        
        # Setup the prompt template
        template = """Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {input}

Answer:"""
        self.prompt = PromptTemplate.from_template(template)

    def process_file(self, file_path: str):
        """Loads a file, splits it into chunks, and adds to FAISS vector store."""
        # Load document
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError("Unsupported file format")
            
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        splits = text_splitter.split_documents(documents)
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            
            # Create QA chain now that we have a retriever
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            combine_docs_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
        else:
            self.vector_store.add_documents(splits)

    def answer_question(self, question: str) -> str:
        """Answers a question based on uploaded documents."""
        if not self.qa_chain:
            return "Please upload some documents first before asking questions."
            
        response = self.qa_chain.invoke({"input": question})
        return response["answer"]
