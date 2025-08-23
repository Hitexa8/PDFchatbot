# PDF_Chat_Rag_App
#  Chat With Your PDF(s)

This is a Streamlit application that allows you to chat with your PDF documents! Upload one or multiple PDF files, and the AI assistant will answer your questions based on the content of those documents, maintaining conversational context.

##  Features

* **PDF Upload**: Easily upload single or multiple PDF documents.
* **Intelligent Q&A**: Ask questions and get answers directly from the content of your uploaded PDFs.
* **🎤 Voice Input**: Record audio directly in your browser and convert speech to text for hands-free interaction.
* **Conversational Memory**: The AI remembers previous turns in the conversation, allowing for natural follow-up questions.
* **Context-Aware Retrieval**: Utilizes a history-aware retriever to reformulate questions based on chat history for better document retrieval.
* **Groq Integration**: Leverages Groq's fast inference for quick AI responses.
* **HuggingFace Embeddings**: Uses HuggingFace's `all-mpnet-base-v2` for efficient text vectorization.

##  Voice Input Feature

The application now supports **browser-based voice recording**:
- 🎙️ Click the microphone button to record your question directly in the browser
- 🔄 Automatic speech-to-text conversion using Google's speech recognition
- 📝 Review transcribed text before sending
- 🌐 No file uploads required - everything happens in real-time

### How to Use Voice Input:
1. Click the "🎤 Use Voice Input" button
2. Click the microphone icon to start recording
3. Speak your question clearly
4. Click "Convert Speech to Text" to transcribe
5. Review the text and click "Send as Message"

##  Setup and Installation

Follow these steps to get your local copy up and running.

### Prerequisites

Before you begin, ensure you have the following:

* **Python 3.8+**
* **Git** (for cloning the repository)
* **Internet Connection** (required for speech recognition service)
* **API Keys**:
    * **Groq API Key**: Required for the language model. You can get one from [Groq](https://console.groq.com/keys).

### 1. Set Up Virtual Environment
```bash
python -m venv myenv
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
```bash
GROQ_API_KEY="your_groq_api_key_here"
```

### 4. Run the Application
```bash
streamlit run app.py
```