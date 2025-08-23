import streamlit as st
from dotenv import load_dotenv
import os
import speech_recognition as sr
import io
import wave
import tempfile
from PIL import Image
import pytesseract
import platform
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS


if platform.system() == "Windows":
    path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = path
PYTESSERACT_AVAILABLE = True
load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")

TEMP_FILE_DIR = "temp_uploaded_files"
os.makedirs(TEMP_FILE_DIR, exist_ok=True)


def extract_text_from_image(image_path):
    """
    Extract text from image using OCR (Optical Character Recognition)
    """
    if not PYTESSERACT_AVAILABLE:
        st.error("❌ **Tesseract OCR is not available**")
        st.info("""
        **To enable image text extraction, please install Tesseract OCR:**
        
        **Windows:**
        1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
        2. Install and add to Windows PATH
        3. Restart your application
        
        **Linux:**
        ```bash
        sudo apt install tesseract-ocr
        pip install pytesseract
        ```
        
        **macOS:**
        ```bash
        brew install tesseract
        pip install pytesseract
        ```
        
        **Alternative:** You can manually type the text content from your image in the chat.
        """)
        return None
    
    try:
        # Check if tesseract is properly configured
        try:
            pytesseract.get_tesseract_version()
        except Exception as tesseract_error:
            st.error("❌ **Tesseract OCR is installed but not properly configured**")
            st.info("""
            **Please ensure Tesseract is properly installed and in your system PATH.**
            
            **Windows users:** You may need to set the tesseract path manually:
            ```python
            pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
            ```
            """)
            return None
        
        # Open image
        image = Image.open(image_path)
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(image)
        
        if not text.strip():
            st.warning("No text could be extracted from this image. The image might not contain readable text.")
            return None
            
        return text
    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.info("💡 **Tip:** Make sure the image contains clear, readable text for better OCR results.")
        return None


def process_image_file(file_path, file_name):
    """
    Process image file and return document-like structure
    """
    text_content = extract_text_from_image(file_path)
    
    if not text_content:
        st.warning(f"⚠️ Could not extract text from image '{file_name}'. This could be because:")
        st.info("""
        - Tesseract OCR is not installed
        - The image doesn't contain readable text
        - The text in the image is not clear enough for OCR
        
        **You can still:**
        - Upload other supported file types (PDF, TXT, DOCX, CSV)
        - Use Normal Chat mode for general conversation
        - Manually describe the image content in your chat
        """)
        return []
    
    # Create a document-like object similar to PDF loader output
    class ImageDocument:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata
    
    document = ImageDocument(
        page_content=text_content,
        metadata={"source": file_name, "file_type": "image"}
    )
    
    return [document]


@st.cache_resource
def get_vectorstore_from_files(file_paths):
    if not file_paths:
        return None
    all_documents = []
    with st.spinner("Processing files and creating knowledge base... This might take a moment."):
        for file_path in file_paths:
            try:
                file_name = os.path.basename(file_path)
                file_extension = os.path.splitext(file_name)[1].lower()
                
                if file_extension == '.pdf':
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    st.info(f"Loaded {len(documents)} pages from PDF '{file_name}'.")
                
                elif file_extension in ['.txt', '.md']:
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents = loader.load()
                    st.info(f"Loaded text file '{file_name}'.")
                
                elif file_extension == '.docx':
                    loader = Docx2txtLoader(file_path)
                    documents = loader.load()
                    st.info(f"Loaded Word document '{file_name}'.")
                
                elif file_extension == '.csv':
                    loader = CSVLoader(file_path)
                    documents = loader.load()
                    st.info(f"Loaded CSV file '{file_name}'.")
                
                elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                    documents = process_image_file(file_path, file_name)
                    if documents:
                        st.info(f"Extracted text from image '{file_name}'.")
                    else:
                        st.warning(f"No text found in image '{file_name}'.")
                        continue
                
                else:
                    st.warning(f"Unsupported file type: {file_extension}")
                    continue
                    
                all_documents.extend(documents)
                
            except Exception as e:
                st.error(f"Error loading '{os.path.basename(file_path)}': {e}")
                continue

        if not all_documents:
            st.warning("No documents could be loaded from the provided files. Please ensure they are valid files.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
        splits = text_splitter.split_documents(all_documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        
        vectorstore = FAISS.from_documents(splits, embeddings)
        st.success("Knowledge base created successfully from uploaded files! You can now ask questions.")
        return vectorstore


def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves or creates a chat history for a given session ID.
    This ensures chat history persists across reruns for a given session.
    """
    if "store" not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def setup_normal_chat():
    """
    Sets up a normal chat chain without RAG functionality.
    """
    llm = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")
    
    normal_chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer questions to the best of your knowledge and be conversational and friendly."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create a simple conversational chain
    from langchain.chains import ConversationChain
    from langchain_core.runnables import RunnableLambda
    
    # Simple chain that just uses the LLM with prompt
    chain = normal_chat_prompt | llm | RunnableLambda(lambda x: {"answer": x.content})
    
    normal_conversational_chain = RunnableWithMessageHistory(
        chain,
        get_chat_history,
        input_messages_key="input",
        output_messages_key="answer",
        history_messages_key="chat_history",
    )
    
    return normal_conversational_chain


def setup_rag_chain(retriever):
    """
    Sets up the RAG chain, including a history-aware retriever and a
    question-answering chain with a general AI assistant prompt.
    """
    llm = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")

    contextualize_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_system_prompt = (
        "You are a helpful AI assistant.Your task is to answer questions based on the provided context only. "
        "Keep your answers concise, detailed and directly relevant to the question. "
        "if the answer is not in the provided context then just say \"the answer is not in the context provided please ask relevant questions\", don't provide the wrong answer or other information."
        "Do not make up information or provide answers from outside the given documents."
        "\n\nContext: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_chat_history,
        input_messages_key="input",
        output_messages_key="answer",
        history_messages_key="chat_history",
    )
    return conversational_rag_chain


def audio_recorder_component():
    if "audio_text" not in st.session_state:
        st.session_state.audio_text = ""
    
    st.write("🎤 **Voice Input**")
    st.info("🎙️ Click the microphone button below to start recording your question")
    
    audio_bytes = st.audio_input("Record your question", key="voice_input")
    
    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")
        
        if st.button("🔄 Convert Speech to Text", type="secondary"):
            with st.spinner("🎧 Converting speech to text..."):
                try:
                    recognizer = sr.Recognizer()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_bytes.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    with sr.AudioFile(tmp_file_path) as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio_data = recognizer.record(source)
                    
                    text = recognizer.recognize_google(audio_data, language='en-US')
                    st.session_state.audio_text = text
                    st.success("✅ Speech converted to text successfully!")
                    
                    os.unlink(tmp_file_path)
                    
                except sr.UnknownValueError:
                    st.error("❌ Could not understand the audio. Please try speaking more clearly.")
                except sr.RequestError as e:
                    st.error(f"❌ Error with speech recognition service: {e}")
                    st.info("💡 Make sure you have an internet connection for speech recognition.")
                except Exception as e:
                    st.error(f"❌ Error processing audio: {e}")
                    st.info("💡 Please try recording again with clear pronunciation.")
                finally:
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
    
    if st.session_state.audio_text:
        st.write("📝 **Transcribed Text:**")
        st.text_area(
            "Your speech was converted to:",
            value=st.session_state.audio_text,
            height=80,
            key="transcribed_text",
            disabled=True
        )
        
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("✅ Send as Message", type="primary", key="use_audio_text"):
                return st.session_state.audio_text
        with col_b:
            if st.button("🗑️ Clear", key="clear_audio_text"):
                st.session_state.audio_text = ""
                st.rerun()
    
    return None


def main():
    st.set_page_config(page_title="Chat With Your Documents 💬", layout="wide")
    st.title("Chat With Your Documents 💬")

    with st.sidebar:
        st.header("Configuration")
        
        # Chat mode selection
        chat_mode = st.radio(
            "Select Chat Mode:",
            ["Document Chat", "Normal Chat"],
            help="Choose between document-based chat or normal conversation with the AI"
        )
        st.write("---")
        
        session_id = st.text_input(
            "Session ID",
            value="default_session",
            help="Enter a unique ID for your chat session to maintain history."
        )
        st.write("---")

        if chat_mode == "Document Chat":
            st.subheader("Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload PDF files, Images, Text files, Word docs, or CSV files",
                type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "gif", "txt", "md", "docx", "csv"],
                accept_multiple_files=True,
                help="Upload documents you want the AI to analyze. Supports PDFs, images (JPG, PNG, etc.), text files, Word documents, and CSV files."
            )
            
            file_paths = []
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(TEMP_FILE_DIR, uploaded_file.name)
                    if not os.path.exists(file_path):
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                st.sidebar.success(f"Successfully staged {len(file_paths)} file(s) for processing.")
            else:
                st.sidebar.info("Please upload documents to enable the chat functionality.")
        else:
            st.subheader("Normal Chat Mode")
            st.info("You are in normal chat mode. You can have a conversation with the AI without document context.")
            file_paths = []
        
        st.write("---")
        if st.button("Clear Chat History", help="Click to clear all messages and start a new conversation."):
            if "store" in st.session_state:
                st.session_state.store[session_id] = ChatMessageHistory()
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
    for msg in chat_history:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    # Handle both text and audio input first (to process any pending input)
    user_input = None
    
    # Check if there's pending audio text to process
    if hasattr(st.session_state, 'pending_audio_text') and st.session_state.pending_audio_text:
        user_input = st.session_state.pending_audio_text
        st.session_state.pending_audio_text = None  # Clear after use

    # Process user input if available
    if user_input:
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

    # Input controls at the bottom
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Regular text input
        text_input = st.chat_input("💬 Type your question here...")
        if text_input:
            st.session_state.pending_text_input = text_input
            st.rerun()
    
    with col2:
        # Voice input toggle
        if st.button("🎤 Use Voice Input", help="Click to switch to voice input mode"):
            st.session_state.show_voice_input = True

    # Voice input section
    if st.session_state.get('show_voice_input', False):
        with st.container():
            st.markdown("### 🎤 Voice Input Mode")
            audio_text = audio_recorder_component()
            if audio_text:
                st.session_state.pending_audio_text = audio_text
                st.session_state.show_voice_input = False
                st.rerun()
            
            # Close voice input mode
            if st.button("❌ Close Voice Input"):
                st.session_state.show_voice_input = False
                st.rerun()

    # Handle text input from chat_input
    if hasattr(st.session_state, 'pending_text_input') and st.session_state.pending_text_input:
        st.session_state.pending_audio_text = st.session_state.pending_text_input
        st.session_state.pending_text_input = None
        st.rerun()


if __name__ == "__main__":
    main()