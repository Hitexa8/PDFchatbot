import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Import custom modules
from document_processing import get_vectorstore_from_files
from chat_management import get_chat_history, clear_chat_history
from chat_chains import setup_normal_chat, setup_rag_chain
from ui_components import (
    render_api_status, render_chat_mode_selection, render_session_input,
    render_file_upload_interface, render_normal_chat_info, render_clear_history_button,
    handle_file_staging, render_chat_history
)

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

TEMP_FILE_DIR = "temp_uploaded_files"
os.makedirs(TEMP_FILE_DIR, exist_ok=True)


def main():
    st.set_page_config(page_title="Chat With Your Documents 💬", layout="wide")
    st.title("Chat With Your Documents 💬")

    with st.sidebar:
        st.header("Configuration")
        
        # API Status indicator
        render_api_status()
        st.write("---")
        
        # Chat mode selection
        chat_mode = render_chat_mode_selection()
        st.write("---")
        
        session_id = render_session_input()
        st.write("---")

        if chat_mode == "Document Chat":
            uploaded_files = render_file_upload_interface()
            file_paths = handle_file_staging(uploaded_files, TEMP_FILE_DIR)
        else:  # Normal Chat mode
            render_normal_chat_info()
            file_paths = []
        
        st.write("---")
        if render_clear_history_button(session_id):
            clear_chat_history(session_id)
            st.success("Chat history cleared!")
            st.rerun()
    
    # Handle different chat modes
    if chat_mode == "Document Chat":
        vectorstore = None
        if file_paths:
            vectorstore = get_vectorstore_from_files(file_paths)
        
        if not vectorstore:
            st.info("Upload documents and wait for the knowledge base to be created to begin interaction.")
            st.stop()
            
        retriever = vectorstore.as_retriever()
        conversational_chain = setup_rag_chain(retriever)
    else:  # Normal Chat mode
        conversational_chain = setup_normal_chat()

    chat_history = get_chat_history(session_id).messages
    render_chat_history(chat_history)

    # Input controls at the bottom
    st.markdown("---")
    
    # Regular text input
    text_input = st.chat_input("💬 Type your question here...")
    if text_input:
        user_input = text_input
        
        # Process user input
        with st.chat_message("human"):
            st.markdown(user_input)
            
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                try:
                    response = conversational_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )
                    st.markdown(response["answer"])
                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")
                    st.warning("Please try again or re-upload the documents if the issue persists.")


if __name__ == "__main__":
    main()