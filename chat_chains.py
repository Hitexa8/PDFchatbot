"""
Chat Chains Module

This module handles the creation and management of different chat chains for RAG and normal conversations.
Provides both document-aware (RAG) and general conversation capabilities using language models.

Models Used:
- Groq Gemma2-9B-IT: Primary language model for chat responses
  - Why: Fast inference, good reasoning capabilities, cost-effective
  - Alternatives: OpenAI GPT-4/3.5, Anthropic Claude, Llama models, Mistral
- LangChain Retrieval Chain: RAG implementation for document-aware conversations
  - Why: Combines retrieval and generation for accurate document-based responses
  - Alternatives: Custom RAG implementations, LlamaIndex, Haystack
- LangChain History-Aware Retriever: Context-aware document retrieval
  - Why: Considers chat history for better retrieval relevance
  - Alternatives: Simple retrieval without history, reranking models
"""

import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from chat_management import get_chat_history


def setup_normal_chat():
    """
    Sets up a normal chat chain without RAG functionality.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")
    
    normal_chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer questions to the best of your knowledge and be conversational and friendly."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create a simple conversational chain
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
    groq_api_key = os.getenv("GROQ_API_KEY")
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
