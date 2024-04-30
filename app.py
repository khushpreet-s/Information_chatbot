import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain import conversational_rag_chain
import pickle

# Load pickled objects
with open('chatbot.pkl', 'rb') as f:
    vectorstore, chain = pickle.load(f)

# Streamlit app
st.title("PDF Chatbot")

query = st.text_input("Enter your query:")
if st.button("Search"):
    if query:
        docs = vectorstore.similarity_search(query)
        # Invoke LangChain model
        response = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": "abc123"}}
        )["answer"]
        if response:
            st.write("Response:")
            st.write(response)
        else:
            st.write("No relevant information found.")
    else:
        st.write("Please enter a query.")