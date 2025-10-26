"""
Part 1 - Web Search implementation using LangGraph with Tracing.

This implementation focuses on:
- Setting up web search using Tavily
- Processing search results
- Formatting responses with citations
- Adding Opik Tracing for observability
"""

from typing import Any, Dict, List, Optional, TypedDict, Tuple
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from opik import configure 
from opik.integrations.langchain import OpikTracer 

import os



class State(TypedDict):
    """State for the query understanding graph."""
    question: str
    response: str
    search_results: Optional[str]
    chat_history: Optional[List[Dict[str, str]]]
    category: Optional[str]
    search_iteration: int
    accumulated_results: List[Any]
class WebSearchChat(ChatInterface):
    """Week 2 Part 1 implementation for web search using LangGraph + Tracing."""
    
    def __init__(self):
        opnai_api_key = os.getenv("OPENAI_API_KEY")
        if not opnai_api_key:
          raise ValueError("OPENAI_API_KEY environment variable is required")
        # self.tracer = OpikTracer() 
        self.llm = ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=0.0,
            api_key=opnai_api_key,
            # callbacks=[self.tracer]
        )
        # Read Tavily API key from environment and pass it to the tool.
        # The Tavily client validates that a key is provided and will raise a
        # helpful ValidationError if it's missing. We surface a clearer
        # message here so students know what to set.
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError(
                "TAVILY_API_KEY environment variable is required for web search;"
                " please set it or pass `tavily_api_key` when constructing TavilySearch."
            )

        search_tool = TavilySearch(
            tavily_api_key=tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
        )
        self.llm.bind_tools([search_tool])
        # keep a direct reference to the tool so we can invoke it from a graph node
        self.search_tool = search_tool
        self.graph = self._build_graph()


    def _ask_question(self, state: State) -> State:
        prompt = PromptTemplate.from_template(
            "You are an AI assistant helping users find accurate information. You will be followed by web searches that will provide additional context and citations.\n\n" \
            "For the following question, provide an initial response that:\n" \
            "1. Identifies key concepts and aspects that need to be researched\n" \
            "2. Forms a clear hypothesis or preliminary answer based on your knowledge\n" \
            "3. Indicates what specific information you would want to verify through search\n\n" \
            "Question: {Question}\n\n" \
            "Remember that your response will be enhanced by subsequent web searches and a final summary with citations, so focus on laying a strong foundation for the research."
        ).invoke({"Question": state["question"]})
        response = self.llm.invoke(prompt)
        return { **state, "response": response.content }
    
    
    def _format_search_results(self, results: List[Any]) -> str:
        """Format search results with citations for summary generation."""
        formatted_results = []
        for i, search_result in enumerate(results, 1):
            if isinstance(search_result, dict) and "results" in search_result:
                for j, result in enumerate(search_result["results"], 1):
                    source = f"[{i}.{j}] {result.get('url', 'No URL')}"
                    content = result.get('content', '')
                    formatted_results.append(f"Source {source}:\n{content}\n")
        return "\n".join(formatted_results)

    def _generate_summary(self, state: State) -> State:
        """Generate a summary from accumulated search results with citations."""
        accumulated = state.get("accumulated_results", [])
        if not accumulated:
            return state
            
        formatted_results = self._format_search_results(accumulated)
        prompt = PromptTemplate.from_template(
            "You have performed multiple web searches and received several results. "
            "Please synthesize a comprehensive answer based on all findings. "
            "Include citations to your sources using the provided source numbers in square brackets "
            "(e.g., [1.1], [2.3], etc.). Each citation should back up the specific claim or information "
            "you're presenting. Make sure to cite multiple sources where appropriate.\n\n"
            "Here are the search results with their sources:\n\n{results}"
        ).invoke({"results": formatted_results})
        
        summary = self.llm.invoke(prompt)
        return {**state, "response": summary.content}

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("ask_question", self._ask_question)
        workflow.add_node("call_search", self._call_search_tool)
        workflow.add_node("generate_summary", self._generate_summary)
        
        # Define condition for continuing search or generating summary
        def should_continue_search(state: State) -> str:
            return "call_search" if state.get("search_iteration", 0) < 2 else "generate_summary"
        
        # Build graph with conditional paths
        workflow.add_edge(START, "ask_question")
        workflow.add_edge("ask_question", "call_search")
        workflow.add_conditional_edges("call_search", should_continue_search)
        workflow.add_edge("generate_summary", END)
        
        return workflow.compile()

    def _call_search_tool(self, state: State) -> State:
        """Invoke the TavilySearch tool and attach results to the state.
        Iterates through search twice and accumulates results.
        """
        # Defensive: ensure the tool exists
        if not hasattr(self, "search_tool") or self.search_tool is None:
            return { **state, "response": "Search tool is not available.", "search_results": None }

        try:
            raw = self.search_tool.invoke({"query": state["question"]})
        except Exception as e:
            return { **state, "response": f"Search failed: {e}", "search_results": None }

        # Accumulate results
        accumulated = state.get("accumulated_results", [])
        accumulated.append(raw)
        
        # Update iteration counter
        iteration = state.get("search_iteration", 0) + 1
        
        answer = raw.get("answer") if isinstance(raw, dict) else None
        new_response = answer or state.get("response", "")
        
        return {
            **state,
            "search_results": raw,
            "response": new_response,
            "search_iteration": iteration,
            "accumulated_results": accumulated
        }
    
    def initialize(self) -> None:
        """Initialize components for web search and tracing.
        
        Students should:
        - Initialize the LLM
        - Set up Tavily search tool
        - Create a LangGraph workflow for web search
        - Add Opik tracing for observability
        """
        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, Optional[Any]]:
        """Process a message using web search and record trace.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            Tuple[str, Optional[Any]]: A tuple of (assistant_response, search_results).
                - assistant_response: The assistant's response based on web search results
                - search_results: Raw results returned by the search tool (may be None)
        """
        initial_state = State(
            question=message,
            chat_history=chat_history,
            search_results=None,
            category=None,
            response="",
            search_iteration=0,
            accumulated_results=[]
        )
        result = self.graph.invoke(initial_state)
        response = result.get('response', '')
        return response.strip()
