"""Part 1 - Query Understanding implementation using LangGraph.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type using conditional edges
- Present information professionally
"""

from typing import Dict, List, Optional, TypedDict
import os
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from perplexia_ai.core.chat_interface import ChatInterface


class State(TypedDict):
    """State for the query understanding graph."""
    question: str
    category: Optional[str]
    response: str


class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding using LangGraph."""
    
    def __init__(self):
        """Initialize components for query understanding using LangGraph.
        
        - Initialize the chat model
        - Build a graph with classifier and response nodes
        - Set up conditional edges for routing based on query category
        - Compile the graph
        """
        # Initialize the language model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=api_key
        )
        
        # Build and compile the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for query understanding."""
        # Create the graph
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("general_response", self._handle_general)
        workflow.add_node("greeting_response", self._handle_greeting)
        workflow.add_node("question_response", self._handle_question)
        
        # Add edges
        workflow.add_edge(START, "classifier")
        
        # Add conditional edges based on classification
        workflow.add_conditional_edges(
            "classifier",
            self._route_query,
            {
                "greeting": "greeting_response",
                "question": "question_response", 
                "general": "general_response"
            }
        )
        
        # All response nodes lead to END
        workflow.add_edge("general_response", END)
        workflow.add_edge("greeting_response", END)
        workflow.add_edge("question_response", END)
        
        return workflow.compile()
    
    def _classify_query(self, state: State) -> State:
        """Classify the user query into categories."""
        question = state["question"]
        
        classification_prompt = f"""
        Classify the following user input into one of these categories:
        - "greeting": Simple hellos, hi, how are you, etc.
        - "question": Specific questions seeking information or explanations
        - "general": General conversation, statements, or anything else
        
        User input: "{question}"
        
        Respond with only the category name (greeting, question, or general).
        """
        
        response = self.llm.invoke(classification_prompt)
        category = response.content.strip().lower()
        
        # Ensure we have a valid category
        if category not in ["greeting", "question", "general"]:
            category = "general"
        
        return {**state, "category": category}
    
    def _route_query(self, state: State) -> str:
        """Route to appropriate response handler based on category."""
        return state["category"]
    
    def _handle_greeting(self, state: State) -> State:
        """Handle greeting messages."""
        response = "Hello! I'm your AI assistant. How can I help you today?"
        return {**state, "response": response}
    
    def _handle_question(self, state: State) -> State:
        """Handle specific questions."""
        question = state["question"]
        
        response_prompt = f"""
        The user has asked a specific question. Please provide a helpful, informative, 
        and professional response. Be concise but thorough.
        
        Question: {question}
        
        Provide your response:
        """
        
        response = self.llm.invoke(response_prompt)
        return {**state, "response": response.content}
    
    def _handle_general(self, state: State) -> State:
        """Handle general conversation."""
        message = state["question"]
        
        response_prompt = f"""
        The user has made a general statement or comment. Respond in a friendly, 
        conversational way that keeps the dialogue going.
        
        User message: {message}
        
        Provide your response:
        """
        
        response = self.llm.invoke(response_prompt)
        return {**state, "response": response.content}
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using query understanding with LangGraph.
        
        - Initialize state with the question
        - Invoke the graph
        - Extract and return the response
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 1
            
        Returns:
            str: The assistant's response
        """
        # Initialize state
        initial_state = State(
            question=message,
            category=None,
            response=""
        )
        
        # Invoke the graph
        result = self.graph.invoke(initial_state)
        
        # Return the response
        return result["response"]