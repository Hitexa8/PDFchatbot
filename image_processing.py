"""
Image Processing Module

This module handles image text extraction and description using AI models and OCR.
Provides functionality to extract text from images and generate descriptions for images without text.

Models Used:
- Google Gemini 1.5 Flash: Primary AI model for intelligent text extraction and image description
  - Why: Excellent multimodal capabilities, accurate OCR, and can describe images without text
  - Alternatives: GPT-4 Vision, Claude 3 Vision, Azure Computer Vision
- Tesseract OCR: Fallback for text extraction when Gemini is unavailable
  - Why: Reliable, free, open-source OCR engine
  - Alternatives: Azure OCR, AWS Textract, Google Cloud Vision API
"""

import streamlit as st
import os
import base64
import platform
from PIL import Image
import pytesseract
import google.generativeai as genai


# Configure Tesseract for Windows
if platform.system() == "Windows":
    path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = path
PYTESSERACT_AVAILABLE = True


def encode_image_to_base64(image_path):
    """Convert image to base64 encoding for Gemini API"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None


def extract_text_with_gemini(image_path):
    """
    Extract text from image using Google Gemini Vision model
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("❌ **Gemini API key not found**")
        st.info("""
        **To enable AI-powered image text extraction:**
        
        1. Get a Gemini API key from: https://makersuite.google.com/app/apikey
        2. Add it to your .env file as: `GEMINI_API_KEY=your_api_key_here`
        3. Restart the application
        
        **Alternative:** The app will fall back to Tesseract OCR if available.
        """)
        return None
    
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Open and process image
        image = Image.open(image_path)
        
        # Create prompt for text extraction
        prompt = """
        Please extract all text content from this image. Follow these guidelines:
        
        1. Extract ALL visible text accurately, including:
           - Headers, titles, and headings
           - Body text and paragraphs
           - Lists, bullet points, and numbered items
           - Table contents and labels
           - Captions and annotations
           - Any handwritten text if clearly legible
        
        2. Maintain the logical structure and formatting:
           - Preserve paragraph breaks
           - Keep list items properly formatted
           - Maintain table structure if present
           - Use appropriate spacing between sections
        
        3. For mixed language content (like Hindi-English), extract both languages accurately
        
        4. If the image contains no readable text, respond with "NO_TEXT_FOUND"
        
        5. Do not add any commentary or explanation - just provide the extracted text
        
        Please extract the text now:
        """
        
        # Generate content with the image
        response = model.generate_content([prompt, image])
        
        if response and response.text:
            extracted_text = response.text.strip()
            
            # Check if no text was found
            if extracted_text.upper() == "NO_TEXT_FOUND":
                st.warning("🤖 Gemini AI detected no readable text in this image")
                return None
            
            # Show success message with text preview
            preview = extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
            st.success(f"✅ **Gemini AI successfully extracted text!**")
            st.info(f"📄 **Preview:** {preview}")
            
            return extracted_text
        else:
            st.error("❌ Gemini API returned no response")
            return None
            
    except Exception as e:
        st.error(f"❌ Error with Gemini AI: {e}")
        st.info("💡 Falling back to Tesseract OCR if available...")
        return None


def describe_image_with_gemini(image_path):
    """
    Get detailed description of image content using Gemini when no text is found
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return None
    
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Open and process image
        image = Image.open(image_path)
        
        # Create prompt for image description
        prompt = """
        This image contains no readable text, but please provide a detailed description of what you see in the image. Include:
        
        1. **Main Subject/Content**: What is the primary focus of the image?
        2. **Visual Elements**: Describe objects, people, places, or scenes visible
        3. **Context & Setting**: Where does this appear to be taken? What's the environment?
        4. **Colors & Style**: Notable colors, artistic style, or visual characteristics
        5. **Potential Relevance**: What kind of information or context might this image provide?
        
        Format your response as a comprehensive description that could be useful for someone who cannot see the image. Be detailed but concise.
        
        Start your response with "IMAGE DESCRIPTION:" followed by your detailed analysis.
        """
        
        # Generate content with the image
        response = model.generate_content([prompt, image])
        
        if response and response.text:
            description = response.text.strip()
            
            # Show success message with description preview
            preview = description[:300] + "..." if len(description) > 300 else description
            st.success(f"✅ **Gemini AI created image description!**")
            st.info(f"🖼️ **Description Preview:** {preview}")
            
            return description
        else:
            st.warning("❌ Gemini API couldn't generate image description")
            return None
            
    except Exception as e:
        st.error(f"❌ Error generating image description: {e}")
        return None


def extract_text_from_image(image_path):
    """
    Extract text from image using AI (Gemini) first, then fallback to OCR.
    If no text is found, get image description from Gemini.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # Try Gemini AI first for better results
    if gemini_api_key:
        st.info("🤖 **Using Gemini AI for intelligent text extraction...**")
        gemini_result = extract_text_with_gemini(image_path)
        if gemini_result:
            return gemini_result
        else:
            # No text found, try to get image description instead
            st.info("🖼️ **No text found, generating image description with Gemini AI...**")
            image_description = describe_image_with_gemini(image_path)
            if image_description:
                return image_description
            else:
                st.warning("⚠️ Gemini AI couldn't extract text or describe image, trying Tesseract OCR...")
    else:
        st.info("🔧 **Gemini API key not configured, using Tesseract OCR...**")
    
    # Fallback to Tesseract OCR
    if not PYTESSERACT_AVAILABLE:
        st.error("❌ **Neither Gemini AI nor Tesseract OCR is available**")
        st.info("""
        **To enable image processing, please either:**
        
        **Option 1: Use Gemini AI (Recommended)**
        1. Get a Gemini API key from: https://makersuite.google.com/app/apikey
        2. Add it to your .env file as: `GEMINI_API_KEY=your_api_key_here`
        3. Restart the application
        
        **Benefits of Gemini AI:**
        ✨ Extracts text AND describes images without text
        ✨ Better accuracy with handwritten text and complex layouts
        ✨ Supports multiple languages simultaneously
        ✨ Provides context for images without readable text
        
        **Option 2: Install Tesseract OCR (Text-only)**
        
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
        
        **Alternative:** You can manually describe the image content in the chat.
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
        
        st.info("🔧 **Using Tesseract OCR for text extraction...**")
        
        # Open image
        image = Image.open(image_path)
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(image)
        
        if not text.strip():
            st.warning("⚠️ Tesseract OCR couldn't extract any text from this image")
            st.info("💡 Consider using Gemini AI for better image understanding (including images without text)")
            return None
        
        # Show success message with preview
        preview = text.strip()[:200] + "..." if len(text.strip()) > 200 else text.strip()
        st.success("✅ **Tesseract OCR successfully extracted text!**")
        st.info(f"📄 **Preview:** {preview}")
            
        return text
    except Exception as e:
        st.error(f"❌ Error processing image with Tesseract: {e}")
        st.info("💡 **Tip:** Make sure the image contains clear, readable text for better OCR results.")
        return None
