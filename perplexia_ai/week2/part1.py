"""
Part 1 - Web Search implementation using LangGraph with Tracing.

This implementation focuses on:
- Setting up web search using Tavily
- Processing search results
- Formatting responses with citations
- Adding Opik Tracing for observability
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from opik import configure 
from opik.integrations.langchain import OpikTracer 

import os

def search_tool(query: str) -> str:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    tavily_client = TavilyClient(api_key=tavily_api_key)
    response = tavily_client.search(query)
    return response


class WebSearchChat(ChatInterface):
    """Week 2 Part 1 implementation for web search using LangGraph + Tracing."""
    
    def __init__(self):
        opnai_api_key = os.getenv("OPENAI_API_KEY")
        if not opnai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.tracer = OpikTracer() 
        self.llm = ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=0.0,
            api_key=opnai_api_key,
            callbacks=[self.tracer]
        )
        self.llm.bind_tools([search_tool])
        self.search_tool = search_tool
        self.graph = None
        self.tracer = None
    
    def initialize(self) -> None:
        """Initialize components for web search and tracing.
        
        Students should:
        - Initialize the LLM
        - Set up Tavily search tool
        - Create a LangGraph workflow for web search
        - Add Opik tracing for observability
        """
        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using web search and record trace.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response based on web search results
        """
        return "Not implemented yet. Please implement Week 2 Part 1: Web Search with LangGraph."
