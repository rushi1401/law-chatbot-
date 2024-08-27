import streamlit as st
import logging
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define categories
CATEGORIES = {
    'Civil Law': 'vectorstore_pro/civil_law/',
    'Commercial Law': 'vectorstore_pro/commercial_law/',
    'Corporate Law': 'vectorstore_pro/Corporate_Law/',
    'Criminal Law': 'vectorstore_pro/Criminal_Law/',
    'Family Law': 'vectorstore_pro/family_law/'
}

# Initialize LLM
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

def load_vector_store(category: str):
    if category not in CATEGORIES:
        st.error("Invalid category.")
        return None
    
    db_path = os.path.join(CATEGORIES[category], 'db_faiss')
    
    if not os.path.exists(db_path):
        st.error("Vector store does not exist for the selected category.")
        return None
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def get_response(query: str, retriever):
    # Define the prompt template
    prompt_template = "Based on the context, answer the question: {query}"
    prompt = PromptTemplate(template=prompt_template, input_variables=["query"])
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Generate a response
    response = chain({"query": query})
    return response.get("text", "No response from LLM")

def main():
    st.set_page_config(page_title="Legal Query Application", layout="wide")
    st.title("Legal Query Application")

    # Sidebar for selecting category and inputting query
    st.sidebar.header("Query Options")
    category = st.sidebar.selectbox("Select a Category:", list(CATEGORIES.keys()))
    query = st.sidebar.text_input("Enter your query:")

    if st.sidebar.button("Submit Query"):
        if query:
            # Load vector store for the selected category
            vectorstore = load_vector_store(category)
            if vectorstore:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
                # Get the response
                answer = get_response(query, retriever)
                st.success("Query processed successfully!")
                st.write(f"**Answer:** {answer}")
        else:
            st.error("Please enter a query.")

    # Display a welcome message
    st.write("Welcome to the Legal Query Application. Use the sidebar to select a category and enter your query.")

if __name__ == "__main__":
    main()
