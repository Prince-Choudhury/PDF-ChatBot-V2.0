import streamlit as st  # Import the Streamlit module
from dotenv import load_dotenv  # Import the dotenv module to load environment variables
import pickle  # Import the pickle module to save and load objects
import os  # Import the os module to interact with the operating system
from PyPDF2 import PdfReader  # Import the PdfReader class from the PyPDF2 module to read PDF files
from streamlit_extras.add_vertical_space import add_vertical_space  # Import the add_vertical_space function from the streamlit_extras module to add vertical space between elements
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Import the RecursiveCharacterTextSplitter class from the langchain module to split text into chunks
from langchain.embeddings.openai import OpenAIEmbeddings  # Import the OpenAIEmbeddings class from the langchain module to create embeddings using OpenAI
from langchain.vectorstores import FAISS  # Import the FAISS class from the langchain module to create a vectorstore using FAISS
from langchain.chains import ConversationalRetrievalChain  # Import the ConversationalRetrievalChain class from the langchain module to create a conversational retrieval chain
from langchain.llms import OpenAI  # Import the OpenAI class from the langchain module to create a language model using OpenAI

# Load environment variables
load_dotenv()

def create_new_chat_session():
    # Function to create a new chat session and set it as active
    chat_id = len(st.session_state.chat_sessions) + 1  # Generate a chat id based on the number of chat sessions
    session_key = f"Chat {chat_id}"  # Create a session key using the chat id
    st.session_state.chat_sessions[session_key] = []  # Initialize an empty list for the chat history
    st.session_state.active_session = session_key  # Set the active session to the session key

def initialize_chat_ui():
    # Function to display the chat messages and inputs based on the active session
    if "active_session" in st.session_state:  # Check if there is an active session
        for message in st.session_state.chat_sessions[st.session_state.active_session]:  # Loop through the chat history of the active session
            with st.chat_message(message["role"]):  # Create a chat message with the role of the sender
                st.markdown(message["content"])  # Display the content of the message using markdown

    return st.chat_input("Ask your questions from PDF ")  # Create a chat input for the user and return the value

# Sidebar contents
with st.sidebar:  # Create a sidebar
    st.title('PDF ChatBot - History and Chat button')  # Set the title for the sidebar

    pdf = st.file_uploader("Upload your PDF", type='pdf')  # Create a file uploader for the PDF
    #add_vertical_space(2)  # Add vertical space between elements
    st.write('Crafted with ✨ by Prince Choudhury')  # Add a credit line with a hyperlink


def main():

    st.header("Chat with PDF 📚")  # Set the header for the main page

    if "chat_sessions" not in st.session_state:  # Check if the chat sessions dictionary is initialized
        st.session_state.chat_sessions = {}  # Initialize the chat sessions dictionary

    if "active_session" not in st.session_state:  # Check if there is an active session
        create_new_chat_session()  # Create a new chat session if none exists

    # New Chat button
    if st.sidebar.button("New Chat"):  # Create a button for a new chat session in the sidebar
        create_new_chat_session()  # Create a new chat session if the button is clicked

    # Buttons for previous chat sessions
    for session in st.session_state.chat_sessions:  # Loop through the chat sessions
        if st.sidebar.button(session, key=session):  # Create a button for each chat session in the sidebar with a unique key
            st.session_state.active_session = session  # Set the active session to the selected one



    if pdf is not None:  # Check if the PDF file is uploaded
        pdf_reader = PdfReader(pdf)  # Read the PDF file
        text = ""
        for page in pdf_reader.pages:  # Loop through the pages of the PDF file
            text += page.extract_text()  # Extract the text from each page and append it to the text variable

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)  # Split the text into chunks using the text splitter

        store_name = pdf.name[:-4]  # Get the name of the PDF file without the extension
        st.write(f'{store_name}')  # Write the name of the PDF file

        if os.path.exists(f"{store_name}.pkl"):  # Check if the vectorstore file exists
            with open(f"{store_name}.pkl", "rb") as f:  # Open the vectorstore file in read mode
                vectorstore = pickle.load(f)  # Load the vectorstore from the file
        else:  # If the vectorstore file does not exist
            embeddings = OpenAIEmbeddings()  # Create an embeddings object using OpenAI
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)  # Create a vectorstore from the chunks using the embeddings
            with open(f"{store_name}.pkl", "wb") as f:  # Open the vectorstore file in write mode
                pickle.dump(vectorstore, f)  # Save the vectorstore to the file

        # Chat UI and processing
        llm = OpenAI(temperature=0, max_tokens=1000)  # Create a language model using OpenAI
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())  # Create a conversational retrieval chain using the language model and the vectorstore
        prompt = initialize_chat_ui()  # Initialize the chat UI and get the user input

        if prompt:  # Check if the user input is not empty
            st.session_state.chat_sessions[st.session_state.active_session].append({"role": "user", "content": prompt})  # Append the user input to the chat history
            with st.chat_message("user"):  # Create a chat message with the user role
                st.markdown(prompt)  # Display the user input using markdown
                chat_history = st.session_state.chat_sessions.get(st.session_state.active_session, [])  # Get the chat history for the active session
                result = qa({"question": prompt, "chat_history": [(msg["role"], msg["content"]) for msg in chat_history]})  # Get the answer from the conversational retrieval chain

            full_response = result["answer"]  # Get the answer text

            with st.chat_message("assistant"):  # Create a chat message with the assistant role
                st.markdown(full_response)  # Display the answer using markdown
            st.session_state.chat_sessions[st.session_state.active_session].append(
                {"role": "assistant", "content": full_response})  # Append the answer to the chat history

if __name__ == '__main__':
    main()  # Run the main function
