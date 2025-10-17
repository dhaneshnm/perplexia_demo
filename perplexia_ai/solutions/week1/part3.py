"""Part 3 - Conversation Memory implementation using LangGraph.

This implementation focuses on:
- Maintain context across messages
- Handle follow-up questions
- Use conversation history in responses
"""

from typing import Dict, List, Optional, TypedDict
import os
import datetime
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator


class State(TypedDict):
    """State for the memory-enabled chat graph."""
    question: str
    history: Optional[str]
    category: Optional[str]
    calculation_expression: Optional[str]
    calculation_result: Optional[str]
    response: str


class MemoryChat(ChatInterface):
    """Week 1 Part 3 implementation adding conversation memory using LangGraph."""
    
    def __init__(self):
        """Initialize components for memory-enabled chat using LangGraph.
        
        - Initialize the chat model
        - Build a graph with classifier, response, and tool nodes
        - Include history in prompts for context-aware responses
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
        """Build the LangGraph workflow for memory-enabled chat."""
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
    
    def _format_chat_history(self, chat_history: Optional[List[Dict[str, str]]]) -> str:
        """Format chat history as a string for context."""
        if not chat_history:
            return "No previous conversation history."
        
        formatted_history = []
        for message in chat_history[-10:]:  # Keep last 10 messages
            role = message.get("role", "unknown")
            content = message.get("content", "")
            formatted_history.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_history)
    
    def _classify_query(self, state: State) -> State:
        """Classify the user query into categories with memory context."""
        question = state["question"]
        history = state.get("history", "No previous conversation history.")
        
        classification_prompt = f"""
        Based on the conversation history and current question, classify the user input into one of these categories:
        - "greeting": Simple hellos, hi, how are you, etc.
        - "calculation": Mathematical expressions or calculation requests (e.g., "what is 5 + 3", "calculate 10 * 2")
        - "datetime": Requests for current time or date
        - "question": Specific questions seeking information or explanations
        - "general": General conversation, statements, or anything else
        
        Conversation history:
        {history}
        
        Current user input: "{question}"
        
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
        """Use calculator tool for mathematical operations with memory context."""
        question = state["question"]
        history = state.get("history", "")
        
        # Extract mathematical expression with context
        extraction_prompt = f"""
        Given the conversation history and current user input, extract the mathematical expression 
        that needs to be calculated. Consider any previous calculations or context.
        
        Conversation history:
        {history}
        
        Current user input: "{question}"
        
        Return only the mathematical expression that can be calculated (e.g., "5 + 3", "10 * (2 + 3)", "15 / 3").
        If the user is referring to a previous calculation, use the context to understand what they mean.
        
        Mathematical expression:
        """
        
        response = self.llm.invoke(extraction_prompt)
        expression = response.content.strip()
        
        # Calculate the result
        result = self.calculator.evaluate_expression(expression)
        
        # Format the response with context awareness
        if isinstance(result, str) and result.startswith("Error"):
            final_response = f"I apologize, but I encountered an error while calculating: {result}"
        else:
            # Check if this seems like a follow-up calculation
            if any(word in question.lower() for word in ["that", "it", "result", "answer", "plus", "minus", "times", "divided"]):
                final_response = f"Based on our conversation, calculating {expression} = {result}"
            else:
                final_response = f"The answer is: {expression} = {result}"
        
        return {
            **state, 
            "calculation_expression": expression,
            "calculation_result": str(result),
            "response": final_response
        }
    
    def _get_datetime(self, state: State) -> State:
        """Get current date and time with memory context."""
        history = state.get("history", "")
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if this is a follow-up to previous time requests
        if "time" in history.lower() or "date" in history.lower():
            response = f"The current date and time is now: {current_time}"
        else:
            response = f"The current date and time is: {current_time}"
            
        return {**state, "response": response}
    
    def _handle_greeting(self, state: State) -> State:
        """Handle greeting messages with memory context."""
        history = state.get("history", "")
        
        # Check if we've greeted before
        if "hello" in history.lower() or "hi" in history.lower():
            response = "Hello again! How else can I help you today?"
        else:
            response = "Hello! I'm your AI assistant. I can help with calculations, tell you the current time, and answer questions while remembering our conversation. How can I help you today?"
            
        return {**state, "response": response}
    
    def _handle_question(self, state: State) -> State:
        """Handle specific questions with memory context."""
        question = state["question"]
        history = state.get("history", "")
        
        response_prompt = f"""
        The user has asked a specific question. Use the conversation history to provide context-aware, 
        helpful, and professional response. If the question relates to something discussed earlier, 
        acknowledge that connection.
        
        Conversation history:
        {history}
        
        Current question: {question}
        
        Provide your response:
        """
        
        response = self.llm.invoke(response_prompt)
        return {**state, "response": response.content}
    
    def _handle_general(self, state: State) -> State:
        """Handle general conversation with memory context."""
        message = state["question"]
        history = state.get("history", "")
        
        response_prompt = f"""
        The user has made a general statement or comment. Use the conversation history to respond 
        in a friendly, conversational way that acknowledges previous interactions and keeps the 
        dialogue going naturally.
        
        Conversation history:
        {history}
        
        Current user message: {message}
        
        Provide your response:
        """
        
        response = self.llm.invoke(response_prompt)
        return {**state, "response": response.content}
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message with memory and tools using LangGraph.
        
        - Format chat_history as a string
        - Initialize state with question and history
        - Invoke the graph
        - Extract and return the response
        
        Args:
            message: The user's input message
            chat_history: List of previous chat messages with 'role' and 'content' keys
            
        Returns:
            str: The assistant's response
        """
        # Format chat history
        formatted_history = self._format_chat_history(chat_history)
        
        # Initialize state
        initial_state = State(
            question=message,
            history=formatted_history,
            category=None,
            calculation_expression=None,
            calculation_result=None,
            response=""
        )
        
        # Invoke the graph
        result = self.graph.invoke(initial_state)
        
        # Return the response
        return result["response"]