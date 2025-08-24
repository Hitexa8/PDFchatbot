"""
Chat Management Module

This module handles chat history management and session persistence across application reruns.
Provides functionality to store, retrieve, and manage conversation history for different chat sessions.

Models Used:
- LangChain ChatMessageHistory: In-memory chat history storage
  - Why: Simple, efficient for session-based chat history management
  - Alternatives: Redis for persistent storage, MongoDB, PostgreSQL with chat tables
- LangChain BaseChatMessageHistory: Base interface for chat history management
  - Why: Standardized interface for different storage backends
  - Alternatives: Custom chat history implementations, external chat services
"""

import streamlit as st
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


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


def clear_chat_history(session_id: str):
    """
    Clear chat history for a specific session
    """
    if "store" in st.session_state:
        st.session_state.store[session_id] = ChatMessageHistory()
    return True


def get_all_sessions():
    """
    Get all active session IDs
    """
    if "store" not in st.session_state:
        return []
    return list(st.session_state.store.keys())


def delete_session(session_id: str):
    """
    Delete a specific chat session
    """
    if "store" in st.session_state and session_id in st.session_state.store:
        del st.session_state.store[session_id]
        return True
    return False
