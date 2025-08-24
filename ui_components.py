"""
User Interface Module

This module handles the Streamlit user interface components including sidebar configuration,
file uploads, API status indicators, and main chat interface rendering.

Models Used:
- Streamlit: Web application framework for creating interactive UIs
  - Why: Easy to use, perfect for ML/AI demos, built-in components
  - Alternatives: Gradio, Flask/FastAPI with custom frontend, React/Vue.js
"""

import streamlit as st
import os


def render_api_status():
    """Render API status indicators in sidebar"""
    st.subheader("🔧 API Status")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # Check Groq API
    if groq_api_key:
        st.success("✅ Groq API: Connected")
    else:
        st.error("❌ Groq API: Not configured")
        
    # Check Gemini API
    if gemini_api_key:
        st.success("✅ Gemini AI: Connected (Advanced image processing)")
    else:
        st.warning("⚠️ Gemini AI: Not configured")
        with st.expander("🚀 Setup Gemini AI for Advanced Image Processing"):
            st.info("""
            **Get comprehensive image understanding with Gemini AI:**
            
            1. **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. **Create .env file** in your project folder (if not exists)
            3. **Add this line**: `GEMINI_API_KEY=your_api_key_here`
            4. **Restart the app**
            
            **Advanced Capabilities:**
            ✨ Extracts text from images (better than OCR)
            ✨ **Describes images without text** (NEW!)
            ✨ Handles handwritten text and complex layouts
            ✨ Supports multiple languages simultaneously
            ✨ Provides context for ALL types of images
            ✨ Works with photos, diagrams, charts, artwork, etc.
            
            **Perfect for:**
            📊 Charts and graphs without text
            🖼️ Photos and artwork
            📋 Forms and documents
            🎨 Diagrams and illustrations
            """)
            
            st.success("**With Gemini AI, EVERY image adds value to your knowledge base!**")


def render_chat_mode_selection():
    """Render chat mode selection interface"""
    chat_mode = st.radio(
        "Select Chat Mode:",
        ["Document Chat", "Normal Chat"],
        help="Choose between document-based chat or normal conversation with the AI"
    )
    return chat_mode


def render_session_input():
    """Render session ID input interface"""
    session_id = st.text_input(
        "Session ID",
        value="default_session",
        help="Enter a unique ID for your chat session to maintain history."
    )
    return session_id


def render_file_upload_interface():
    """Render file upload interface for document chat"""
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files, Images, Text files, Word docs, or CSV files",
        type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "gif", "txt", "md", "docx", "csv"],
        accept_multiple_files=True,
        help="Upload documents you want the AI to analyze. Supports PDFs, images (JPG, PNG, etc.), text files, Word documents, and CSV files."
    )
    return uploaded_files


def render_normal_chat_info():
    """Render information for normal chat mode"""
    st.subheader("Normal Chat Mode")
    st.info("You are in normal chat mode. You can have a conversation with the AI without document context.")


def render_clear_history_button(session_id):
    """Render clear chat history button"""
    if st.button("Clear Chat History", help="Click to clear all messages and start a new conversation."):
        return True
    return False


def handle_file_staging(uploaded_files, temp_dir):
    """Handle staging of uploaded files"""
    file_paths = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        st.sidebar.success(f"Successfully staged {len(file_paths)} file(s) for processing.")
    else:
        st.sidebar.info("Please upload documents to enable the chat functionality.")
    return file_paths


def render_chat_history(chat_history):
    """Render existing chat messages"""
    for msg in chat_history:
        with st.chat_message(msg.type):
            st.markdown(msg.content)
