import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Clean & Direct LangChain Imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="RAG PDF AI", layout="wide")
st.title("📄 RAG-Powered PDF Q&A Bot")

api_key = st.sidebar.text_input("Groq API Key:", type="password")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        with st.spinner("Indexing PDF into FAISS Vector DB..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            docs = PyPDFLoader(tmp_path).load()
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer strictly based on context:\n\n{context}"),
                ("human", "{input}")
            ])
            
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)
            st.success("Vector DB ready!")

        user_query = st.text_input("Ask about the PDF:")
        if user_query:
            res = chain.invoke({"input": user_query})
            st.markdown(f"**Answer:** {res['answer']}")
else:
    st.info("Enter Groq API Key to proceed.")
