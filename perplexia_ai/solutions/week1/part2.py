"""Part 2 - Basic Tools implementation using LangGraph.

This implementation focuses on:
- Detect when calculations are needed
- Use calculator for mathematical operations
- Format calculation results clearly
"""

from typing import Dict, List, Optional, TypedDict
import os
import datetime
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator


class State(TypedDict):
    """State for the basic tools graph."""
    question: str
    category: Optional[str]
    calculation_expression: Optional[str]
    calculation_result: Optional[str]
    response: str


class BasicToolsChat(ChatInterface):
    """Week 1 Part 2 implementation adding calculator functionality using LangGraph."""
    
    def __init__(self):
        """Initialize components for basic tools using LangGraph.
        
        - Initialize the chat model
        - Build a graph with classifier, response, and tool nodes
        - Add calculator and datetime tool nodes
        - Set up conditional edges for routing
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
        
        self.calculator = Calculator()
        
        # Build and compile the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for basic tools."""
        # Create the graph
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("calculator_tool", self._use_calculator)
        workflow.add_node("datetime_tool", self._get_datetime)
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
                "calculation": "calculator_tool",
                "datetime": "datetime_tool",
                "question": "question_response", 
                "general": "general_response"
            }
        )
        
        # All response nodes lead to END
        workflow.add_edge("general_response", END)
        workflow.add_edge("greeting_response", END)
        workflow.add_edge("question_response", END)
        workflow.add_edge("calculator_tool", END)
        workflow.add_edge("datetime_tool", END)
        
        return workflow.compile()
    
    def _classify_query(self, state: State) -> State:
        """Classify the user query into categories."""
        question = state["question"]
        
        classification_prompt = f"""
        Classify the following user input into one of these categories:
        - "greeting": Simple hellos, hi, how are you, etc.
        - "calculation": Mathematical expressions or calculation requests (e.g., "what is 5 + 3", "calculate 10 * 2")
        - "datetime": Requests for current time or date
        - "question": Specific questions seeking information or explanations
        - "general": General conversation, statements, or anything else
        
        User input: "{question}"
        
        Respond with only the category name (greeting, calculation, datetime, question, or general).
        """
        
        response = self.llm.invoke(classification_prompt)
        category = response.content.strip().lower()
        
        # Ensure we have a valid category
        if category not in ["greeting", "calculation", "datetime", "question", "general"]:
            category = "general"
        
        return {**state, "category": category}
    
    def _route_query(self, state: State) -> str:
        """Route to appropriate handler based on category."""
        return state["category"]
    
    def _use_calculator(self, state: State) -> State:
        """Use calculator tool for mathematical operations."""
        question = state["question"]
        
        # Extract mathematical expression
        extraction_prompt = f"""
        Extract the mathematical expression from this user input. Return only the mathematical 
        expression that can be calculated (e.g., "5 + 3", "10 * (2 + 3)", "15 / 3").
        
        User input: "{question}"
        
        Mathematical expression:
        """
        
        response = self.llm.invoke(extraction_prompt)
        expression = response.content.strip()
        
        # Calculate the result
        result = self.calculator.evaluate_expression(expression)
        
        # Format the response
        if isinstance(result, str) and result.startswith("Error"):
            final_response = f"I apologize, but I encountered an error while calculating: {result}"
        else:
            final_response = f"The answer is: {expression} = {result}"
        
        return {
            **state, 
            "calculation_expression": expression,
            "calculation_result": str(result),
            "response": final_response
        }
    
    def _get_datetime(self, state: State) -> State:
        """Get current date and time."""
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        response = f"The current date and time is: {current_time}"
        return {**state, "response": response}
    
    def _handle_greeting(self, state: State) -> State:
        """Handle greeting messages."""
        response = "Hello! I'm your AI assistant. I can help with calculations, tell you the current time, and answer questions. How can I help you today?"
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
        """Process a message with calculator support using LangGraph.
        
        - Initialize state with the question
        - Invoke the graph
        - Extract and return the response
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 2
            
        Returns:
            str: The assistant's response
        """
        # Initialize state
        initial_state = State(
            question=message,
            category=None,
            calculation_expression=None,
            calculation_result=None,
            response=""
        )
        
        # Invoke the graph
        result = self.graph.invoke(initial_state)
        
        # Return the response
        return result["response"]