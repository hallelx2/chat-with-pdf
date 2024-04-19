import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores.faiss import FAISS

# how to get the response for the pdf file
def get_response(pdf, user_question):
    load_dotenv()
    # extract text
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator = '\n',
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)

        # Embed the text
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        knowledge_store = FAISS.from_texts(chunks, embeddings)
        if user_question:
            docs = knowledge_store.similarity_search(query=user_question)
            chain = load_qa_chain(llm, chain_type='stuff')
            response = chain.run(question = user_question, input_documents = docs)
            
            return response
            

def main():
    load_dotenv()
    st.set_page_config(page_title = 'Chat with your PDF')
    st.header('Ask your PDF :chat')

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content = "Hello, I can help you find the right answers to the questions about your PDF, How can I help you"),
        ]

    pdf = st.file_uploader('Upload your PDF', type='pdf')


    # User functionality
    user_query = st.chat_input('Ask your question')

    if user_query is not None and user_query!='':
        response = get_response(pdf, user_query)
        st.session_state.chat_history.append(HumanMessage(content= user_query))
        st.session_state.chat_history.append(AIMessage(content = response))
        
    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message('AI'):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message('Human'):
                st.write(message.content)

if __name__=='__main__':
    main()