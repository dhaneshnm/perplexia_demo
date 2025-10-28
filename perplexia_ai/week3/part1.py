"""Part 1 - Tool-Using Agent implementation.

This implementation focuses on:
- Converting tools from Assignment 1 to use with LangGraph
- Using the ReAct pattern for autonomous tool selection
- Comparing manual workflow vs agent approaches
"""

from typing import Any, Dict, List, Optional, TypedDict, Tuple
from perplexia_ai.core.chat_interface import ChatInterface

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
import datetime
import os
import json

@tool
def calculator(expr: str) -> float:
    """Evaluates a mathematical expression and returns the result as a float."""
    try:
        result = eval(expr, {"__builtins__": None}, {})
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {expr}. Error: {e}")


@tool
def get_current_date() -> str:
    """Returns the current date as a string in YYYY-MM-DD format."""
    return datetime.date.today().isoformat()

class ToolUsingAgentChat(ChatInterface):
    """Week 3 Part 1 implementation focusing on tool-using agents with Tavily weather lookup."""
    
    def __init__(self):
        self.llm = None
        self.tools = [calculator, get_current_date]
        self.weather_tool = None
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for the tool-using agent including Tavily weather lookup."""
        opnai_api_key = os.getenv("OPENAI_API_KEY")
        if not opnai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.llm = ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=0.0,
            api_key=opnai_api_key,
        )

        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError(
                "TAVILY_API_KEY environment variable is required for weather search;"
                " please set it or pass `tavily_api_key` when constructing TavilySearch."
            )
        self.weather_tool = TavilySearch(
            tavily_api_key=tavily_api_key,
            max_results=2,
            topic="news",
            include_answer=True,
            include_raw_content=False
        )
        
        # Build the graph after initializing tools and LLM
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the ReAct agent graph using create_react_agent."""
        all_tools = self.tools + [self.weather_tool]
        
        # Create the ReAct agent
        return create_react_agent(self.llm, all_tools)    
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the tool-using agent.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response
        """
        # Create system message with tool usage instructions
        system_message = SystemMessage(content="""You are an assistant that can answer questions and use tools when needed.

Available tools:
- calculator: Evaluates mathematical expressions.
- get_current_date: Returns the current date in YYYY-MM-DD format.
- tavily_search_results_json: Looks up current and forecasted weather information for a location using Tavily web search.

Instructions:
- If the question asks for today's date, always use the get_current_date tool.
- If the question requires any calculation, always use the calculator tool.
- If the question is about weather, location, temperature, forecast, or similar, use the tavily_search_results_json tool to look up the latest weather information and cite sources.
- For all other questions, answer directly without using a tool.
- IMPORTANT: If you decide a tool is needed, ALWAYS call the tool and return its result directly in your answer. Do not just say you will use a tool; always execute the tool and include its output in your response.""")
        
        # Start with system message
        messages = [system_message]
        
        # Convert chat history to message format if provided
        if chat_history:
            for entry in chat_history:
                if 'user' in entry and 'assistant' in entry:
                    messages.append(HumanMessage(content=entry['user']))
                    messages.append(AIMessage(content=entry['assistant']))
        
        # Add the current user message
        messages.append(HumanMessage(content=message))
        
        # Invoke the ReAct agent
        result = self.graph.invoke({"messages": messages})
        
        # Extract the last AI message from the result
        if 'messages' in result and result['messages']:
            last_message = result['messages'][-1]
            if hasattr(last_message, 'content'):
                return last_message.content.strip()
        
        return "I apologize, but I couldn't process your request."

