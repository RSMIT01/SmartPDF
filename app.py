import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings.base import Embeddings
import google.generativeai as genai
import os
import requests
from htmlTemplates import css, bot_template, user_template

API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API with your key
genai.configure(api_key=API_KEY)


def generate_embedding(text: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={API_KEY}"

    payload = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]},
        "taskType": "retrieval_document",
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["embedding"]["values"]


# Create a wrapper to fit LangChain's embedding interface
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [generate_embedding(text) for text in texts]

    def embed_query(self, text):
        return generate_embedding(text)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=True, length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


# Use the custom Gemini embeddings
def get_vector_store(text_chunks):
    embeddings = GeminiEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=API_KEY,
        temperature=0.7,
        convert_system_message_to_human=True,
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your PDFs")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # get pdf
                raw_text = get_pdf_text(pdf_docs)
                # get text chunks
                text_chunks = get_text_chunks(raw_text)
                # create vector store embedding
                vectorstore = get_vector_store(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
