"""
Document Processing Module

This module handles loading, processing, and converting various document formats into vector embeddings.
Supports PDF, text, Word, CSV, and image files for RAG (Retrieval Augmented Generation) applications.

Models Used:
- HuggingFace all-mpnet-base-v2: Text embedding model for semantic search
  - Why: High-quality sentence embeddings, good for semantic similarity tasks
  - Alternatives: OpenAI text-embedding-ada-002, Sentence-BERT models, E5 embeddings
- FAISS (Facebook AI Similarity Search): Vector database for efficient similarity search
  - Why: Fast, memory-efficient, supports large-scale vector operations
  - Alternatives: Pinecone, Weaviate, Chroma, Qdrant
- RecursiveCharacterTextSplitter: Text chunking for better retrieval
  - Why: Preserves document structure while creating optimal chunk sizes
  - Alternatives: TokenTextSplitter, SpacyTextSplitter, SemanticSplitter
"""

import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from image_processing import extract_text_from_image


def process_image_file(file_path, file_name):
    """
    Process image file and return document-like structure with text or description
    """
    content = extract_text_from_image(file_path)
    
    if not content:
        st.warning(f"⚠️ Could not process image '{file_name}'. This could be because:")
        
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key and not hasattr(st.session_state, 'PYTESSERACT_AVAILABLE'):
            st.info("""
            **No image processing method available!**
            
            **Recommended: Use Gemini AI for comprehensive image understanding**
            1. Get a Gemini API key from: https://makersuite.google.com/app/apikey
            2. Add it to your .env file as: `GEMINI_API_KEY=your_api_key_here`
            3. Restart the application
            
            **Gemini AI Benefits:**
            ✨ Extracts text from images
            ✨ Describes images without text
            ✨ Handles complex layouts and multiple languages
            ✨ Provides context for all types of images
            
            **Alternative: Install Tesseract OCR (Text-only)**
            - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
            - Linux: `sudo apt install tesseract-ocr`
            - macOS: `brew install tesseract`
            """)
        elif not gemini_api_key:
            st.info("""
            **Currently using Tesseract OCR (Text extraction only)**
            
            **Upgrade to Gemini AI for comprehensive image understanding:**
            1. Get a Gemini API key from: https://makersuite.google.com/app/apikey
            2. Add it to your .env file as: `GEMINI_API_KEY=your_api_key_here`
            3. Restart the application
            
            **With Gemini AI you get:**
            ✨ Text extraction AND image descriptions
            ✨ Better accuracy with handwritten text
            ✨ Support for images without readable text
            ✨ Multiple language support
            ✨ Context understanding for all image types
            """)
        else:
            st.info("""
            **The image couldn't be processed by Gemini AI**
            
            This might happen due to:
            - API connectivity issues
            - Image format problems
            - Temporary service unavailability
            
            **You can still:**
            - Try uploading the image again
            - Upload other supported file types (PDF, TXT, DOCX, CSV)
            - Use Normal Chat mode for general conversation
            - Manually describe the image content in your chat
            """)
        return []
    
    # Determine content type
    content_type = "text" if not content.startswith("IMAGE DESCRIPTION:") else "description"
    
    # Create a document-like object similar to PDF loader output
    class ImageDocument:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata
    
    # Add metadata about the content type
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    metadata = {
        "source": file_name, 
        "file_type": "image",
        "content_type": content_type,
        "processing_method": "gemini_ai" if gemini_api_key else "tesseract_ocr"
    }
    
    document = ImageDocument(
        page_content=content,
        metadata=metadata
    )
    
    # Provide feedback about what was processed
    if content_type == "description":
        st.info(f"🖼️ **Added image description to knowledge base**: '{file_name}'")
        st.success("✅ This image will provide visual context for your conversations!")
    else:
        st.info(f"📄 **Added extracted text to knowledge base**: '{file_name}'")
        st.success("✅ This text content is now available for document chat!")
    
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
                        # Success feedback is already provided by process_image_file function
                        pass
                    else:
                        st.warning(f"Could not process image '{file_name}' - skipping.")
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
