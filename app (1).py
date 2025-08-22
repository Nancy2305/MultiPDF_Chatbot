import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# ✅ Use local model via HuggingFacePipeline instead of HuggingFaceHub
def get_llm():
    pipe = pipeline(
        "text2text-generation",  # or "text-generation" depending on model
        model="google/flan-t5-base",  # this downloads the model locally
        max_length=512,
        temperature=0.5
    )
    return HuggingFacePipeline(pipeline=pipe)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = get_llm()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            content = message[0] if isinstance(message, tuple) else message.content
            st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
        else:
            content = message[0] if isinstance(message, tuple) else message.content
            st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat With Multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat With Multiple PDFs :books:")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation is not None:
        handle_user_input(user_question)
    elif user_question:
        st.warning("Please upload and process documents first.")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):

                st.info("Extracting text from documents...")
                raw_text = get_pdf_text(pdf_docs)

                st.info("Splitting text into chunks...")
                text_chunks = get_text_chunks(raw_text)

                st.info("Generating embeddings and creating vectorstore...")
                vectorstore = get_vectorstore(text_chunks)

                st.info("Setting up conversation chain...")
                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("PDFs processed successfully. You can now ask questions.")


if __name__ == '__main__':
    main()
