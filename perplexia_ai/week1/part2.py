from typing import Dict, List, Optional, TypedDict
from langchain_core.prompts import PromptTemplate
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator
import os

class State(TypedDict):
    """State for the query understanding graph."""
    question: str
    chat_history: Optional[List[Dict[str, str]]]
    category: Optional[str]
    response: str

class BasicToolsChat(ChatInterface):
    def _format_chat_history(self, chat_history: Optional[List[Dict[str, str]]]) -> str:
        """Format chat history for prompt context."""
        if not chat_history:
            return ""
        history_str = ""
        for turn in chat_history:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            history_str += f"{role.capitalize()}: {content}\n"
        return history_str
    """Week 1 Part 1 implementation focusing on query understanding using LangGraph."""
    
    def __init__(self):
        """Initialize components for query understanding using LangGraph.
        
        Students should:
        - Initialize the chat model
        - Build a graph with classifier and response nodes
        - Set up conditional edges for routing based on query category
        - Compile the graph
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.llm = ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=0.0,
            api_key=api_key
        )
        self.graph = self._build_graph()
        self.calculator = Calculator()
        self.llm.bind_tools([self.calculator.evaluate_expression])

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(State)
        # Add classifier node
        workflow.add_node("classifier", self.classifier)
        # Add nodes for each category
        workflow.add_node("ask_factual_question", self._ask_factual_question)
        workflow.add_node("ask_analytical_question", self._ask_analytical_question)
        workflow.add_node("ask_comparison_question", self._ask_comparison_question)
        workflow.add_node("ask_definition_question", self._ask_definition_question)
        workflow.add_node("ask_calculation_question", self._ask_calculation_question)
        workflow.add_node("ask_datetime_question", self._ask_datetime_question)

        # Edges: Start -> classifier
        workflow.add_edge(START, "classifier")
        # Conditional edges based on classifier output
        workflow.add_conditional_edges(
            "classifier",
            lambda state: state["category"].strip().lower(),
            {
                "factual": "ask_factual_question",
                "analytical": "ask_analytical_question",
                "comparison": "ask_comparison_question",
                "definition": "ask_definition_question",
                "calculation": "ask_calculation_question",
                "datetime": "ask_datetime_question"
            }
        )
        # All category nodes go to END
        workflow.add_edge("ask_factual_question", END)
        workflow.add_edge("ask_analytical_question", END)
        workflow.add_edge("ask_comparison_question", END)
        workflow.add_edge("ask_definition_question", END)
        workflow.add_edge("ask_calculation_question", END)
        workflow.add_edge("ask_datetime_question", END)
        return workflow.compile()
    def _ask_calculation_question(self, state: State) -> State:
        calculator_template = PromptTemplate.from_template("Answer this quetion using calculator tool: {question}")
        prompt = calculator_template.invoke({"question" : state["question"]})
        response = self.llm.invoke(prompt)
        return { **state, "response": response.content }
    def _ask_datetime_question(self, state: State) -> State:
        from perplexia_ai.tools.datetime_tool import DateTimeTool
        # Use DateTimeTool to answer the question
        answer = DateTimeTool.answer_datetime(state["question"])
        return { **state, "response": answer }
    
    def classifier(self, state: State) -> State:
        classifier_template = PromptTemplate.from_template(
            "You are an expert question classifier. Read the following question and assign it to one of these categories: Factual, Analytical, Comparison, Definition, Calculation, or DateTime.\n"
            "Here are examples for each category:\n"
            "Factual: 'What is the capital of France?'\n"
            "Analytical: 'Why did the Roman Empire fall?'\n"
            "Comparison: 'How does Python differ from Java?'\n"
            "Definition: 'What is photosynthesis?'\n"
            "Calculation: 'What is 23 + 19?'\n"
            "DateTime: 'What is the date today?' or 'When is the next solar eclipse?'\n"
            "Question: {question}\n"
            "Respond with only the category name."
        )
        prompt = classifier_template.invoke({"question" : state["question"]})
        response = self.llm.invoke(prompt)
        return { **state, "category": response.content }


    def _ask_factual_question(self, state: State) -> State:
        history_str = self._format_chat_history(state.get("chat_history"))
        prompt = PromptTemplate.from_template(
            "{history}Answer the following factual question concisely and accurately.\n"
            "Example: Q: What is the capital of France?\nA: Paris.\n"
            "Question: {question}\nA:"
        ).invoke({"question": state["question"], "history": history_str})
        response = self.llm.invoke(prompt)
        return { **state, "response": response.content }
    def _ask_analytical_question(self, state: State) -> State:
        history_str = self._format_chat_history(state.get("chat_history"))
        prompt = PromptTemplate.from_template(
            "{history}Provide a thoughtful and well-reasoned answer to the following analytical question.\n"
            "Example: Q: Why did the Roman Empire fall?\nA: The Roman Empire fell due to a combination of internal weaknesses, economic troubles, and external invasions.\n"
            "Question: {question}\nA:"
        ).invoke({"question": state["question"], "history": history_str})
        response = self.llm.invoke(prompt)
        return { **state, "response": response.content }
    def _ask_comparison_question(self, state: State) -> State:
        history_str = self._format_chat_history(state.get("chat_history"))
        prompt = PromptTemplate.from_template(
            "{history}Compare the items or concepts in the following question, highlighting similarities and differences. Present your answer in a markdown table for clarity.\n"
            "Example: Q: How does Python differ from Java?\n"
            "A: | Feature         | Python                        | Java                          |\n"
            "|------------------|-------------------------------|-------------------------------|\n"
            "| Typing           | Dynamically typed             | Statically typed              |\n"
            "| Execution        | Interpreted                   | Compiled                      |\n"
            "| Syntax           | Concise, readable             | More verbose                  |\n"
            "| Use Cases        | Rapid prototyping, scripting  | Enterprise applications        |\n"
            "| Popularity       | Widely used in data science   | Widely used in large systems  |\n"
            "Question: {question}\nA:"
        ).invoke({"question": state["question"], "history": history_str})
        response = self.llm.invoke(prompt)
        return { **state, "response": response.content }
    def _ask_definition_question(self, state: State) -> State:
        history_str = self._format_chat_history(state.get("chat_history"))
        prompt = PromptTemplate.from_template(
            "{history}Provide a clear and concise definition for the following term or concept.\n"
            "Example: Q: What is photosynthesis?\nA: Photosynthesis is the process by which green plants convert sunlight into chemical energy.\n"
            "Question: {question}\nA:"
        ).invoke({"question": state["question"], "history": history_str})
        response = self.llm.invoke(prompt)
        return { **state, "response": response.content }
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using query understanding with LangGraph.
        
        Students should:
        - Initialize state with the question
        - Invoke the graph
        - Extract and return the response
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 1
            
        Returns:
            str: The assistant's response
        """
        # TODO: Students implement query understanding using LangGraph
        initial_state = State(
            question=message,
            chat_history=chat_history,
            category=None,
            response=""
        )
        result = self.graph.invoke(initial_state)
        return f"Category: {result['category'].strip()}\nResponse: {result['response'].strip()}"

