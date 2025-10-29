"""Part 3 - Deep Research Multi-Agent System implementation.

This implementation focuses on:
- Creating specialized agents for different research tasks
- Coordinating multiple agents for comprehensive research
- Generating structured research reports
- Managing complex multi-agent workflows
"""

import os
import json
from typing import Dict, List, Optional, TypedDict, Any
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END


class DeepResearchState(TypedDict):
    """State for the deep research multi-agent system."""
    topic: str                           # Original research topic
    research_plan: Optional[Dict]        # Generated research plan with sections
    current_section_index: int           # Index of current section being researched
    completed_sections: Dict[str, str]   # section_id -> completed research content
    final_report: Optional[str]          # Complete formatted report
    response: str                        # Final response to return


class DeepResearchChat(ChatInterface):
    """Week 3 Part 3 implementation focusing on multi-agent deep research."""
    
    def __init__(self):
        self.llm = None
        self.search_tool = None
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for the deep research system.
        
        Implementation includes:
        - Initialize the chat model
        - Create Tavily search tool for web research
        - Build a multi-agent workflow using LangGraph
        - Implement coordination between agents
        """
        # Initialize OpenAI components
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        self.llm = ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=0.1,  # Lower temperature for more consistent research output
            api_key=openai_api_key,
        )
        
        # Initialize Tavily web search tool
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
            
        self.search_tool = TavilySearch(
            tavily_api_key=tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False
        )
        
        # Build the multi-agent workflow graph
        self.graph = self._build_graph()
    
    def _research_manager(self, state: DeepResearchState) -> DeepResearchState:
        """Research Manager Agent: Creates research plan and report structure."""
        
        # First, do initial web search to understand the topic scope
        try:
            initial_search = self.search_tool.invoke({"query": state["topic"]})
        except Exception as e:
            print(f"Error in initial search: {e}")
            initial_search = {"results": []}
        
        # Create research plan based on initial search results
        planning_prompt = PromptTemplate.from_template("""
You are a Research Manager Agent tasked with creating a comprehensive research plan.

Topic: {topic}

Initial search results: {search_results}

Create a structured research plan with the following format:
1. Identify 3-5 specific research questions/sections for detailed analysis
2. For each section, provide:
   - section_id: A unique identifier (e.g., "section_1", "section_2")
   - title: Clear section title
   - research_prompt: Detailed description of what should be researched
   - search_queries: 2-3 specific search queries to find relevant information

Output your plan as a JSON object with this structure:
{{
    "sections": [
        {{
            "section_id": "section_1",
            "title": "Section Title Here",
            "research_prompt": "Detailed description of what to research in this section...",
            "search_queries": ["query 1", "query 2", "query 3"]
        }}
    ]
}}

Focus on creating sections that will provide comprehensive coverage of the topic.
Each section should explore different aspects, perspectives, or components of the topic.
""")
        
        # Format search results for the prompt
        search_results_text = ""
        if isinstance(initial_search, dict) and "results" in initial_search:
            search_results_text = json.dumps(initial_search["results"][:3], indent=2)
        
        prompt = planning_prompt.invoke({
            "topic": state["topic"],
            "search_results": search_results_text
        })
        
        # Get research plan from LLM
        response = self.llm.invoke(prompt)
        
        try:
            # Parse the JSON response
            plan_content = response.content.strip()
            if plan_content.startswith("```json"):
                plan_content = plan_content[7:-3].strip()
            elif plan_content.startswith("```"):
                plan_content = plan_content[3:-3].strip()
            
            research_plan = json.loads(plan_content)
        except json.JSONDecodeError as e:
            print(f"Error parsing research plan JSON: {e}")
            # Fallback plan
            research_plan = {
                "sections": [
                    {
                        "section_id": "section_1",
                        "title": f"Overview of {state['topic']}",
                        "research_prompt": f"Provide a comprehensive overview of {state['topic']}",
                        "search_queries": [state["topic"], f"{state['topic']} overview", f"{state['topic']} definition"]
                    }
                ]
            }
        
        return {
            **state,
            "research_plan": research_plan,
            "current_section_index": 0,
            "completed_sections": {}
        }
    
    def _specialized_research_agent(self, state: DeepResearchState) -> DeepResearchState:
        """Specialized Research Agent: Conducts detailed research for current section."""
        
        if not state.get("research_plan") or not state["research_plan"].get("sections"):
            return {**state, "response": "No research plan available"}
        
        sections = state["research_plan"]["sections"]
        current_index = state.get("current_section_index", 0)
        
        if current_index >= len(sections):
            return state  # No more sections to research
        
        current_section = sections[current_index]
        section_id = current_section["section_id"]
        research_prompt = current_section["research_prompt"]
        search_queries = current_section.get("search_queries", [state["topic"]])
        
        # Conduct multiple web searches for this section
        all_search_results = []
        for query in search_queries[:3]:  # Limit to 3 queries to control costs
            try:
                search_result = self.search_tool.invoke({"query": query})
                all_search_results.append({
                    "query": query,
                    "results": search_result
                })
            except Exception as e:
                print(f"Error searching for '{query}': {e}")
                continue
        
        # Synthesize findings from all searches
        synthesis_prompt = PromptTemplate.from_template("""
You are a Specialized Research Agent. Your task is to synthesize information from multiple web searches into a comprehensive section for a research report.

Section Title: {section_title}
Research Objective: {research_prompt}

Search Results:
{search_results}

Instructions:
1. Analyze all the search results provided
2. Extract key information relevant to the research objective
3. Synthesize the findings into a well-structured section
4. Include proper citations using [Source X] format where X is the result number
5. Ensure the content directly addresses the research objective
6. Make the section comprehensive but focused

Write a detailed section (3-5 paragraphs) that covers the research objective thoroughly.
Include specific facts, statistics, examples, and insights from the search results.
End with a brief summary of the key points covered in this section.
""")
        
        # Format search results for synthesis
        formatted_results = []
        source_counter = 1
        for search_data in all_search_results:
            query = search_data["query"]
            results = search_data.get("results", {})
            
            if isinstance(results, dict) and "results" in results:
                for result in results["results"][:2]:  # Top 2 results per query
                    formatted_results.append(f"[Source {source_counter}] Query: '{query}'\nTitle: {result.get('title', 'N/A')}\nContent: {result.get('content', 'N/A')}\nURL: {result.get('url', 'N/A')}\n")
                    source_counter += 1
        
        search_results_text = "\n".join(formatted_results)
        
        synthesis_prompt_filled = synthesis_prompt.invoke({
            "section_title": current_section["title"],
            "research_prompt": research_prompt,
            "search_results": search_results_text
        })
        
        # Generate the section content
        section_response = self.llm.invoke(synthesis_prompt_filled)
        section_content = section_response.content.strip()
        
        # Update state with completed section
        completed_sections = state.get("completed_sections", {}).copy()
        completed_sections[section_id] = {
            "title": current_section["title"],
            "content": section_content
        }
        
        return {
            **state,
            "completed_sections": completed_sections,
            "current_section_index": current_index + 1
        }
    
    def _should_continue_research(self, state: DeepResearchState) -> str:
        """Conditional edge: Determine whether to continue research or move to finalizer."""
        if not state.get("research_plan") or not state["research_plan"].get("sections"):
            return "finalizer"
        
        total_sections = len(state["research_plan"]["sections"])
        current_index = state.get("current_section_index", 0)
        
        if current_index < total_sections:
            return "research_agent"  # Continue researching
        else:
            return "finalizer"  # All sections completed, move to finalizer
    
    def _finalizer(self, state: DeepResearchState) -> DeepResearchState:
        """Finalizer: Generates Executive Summary, Key Findings, and Limitations."""
        
        completed_sections = state.get("completed_sections", {})
        if not completed_sections:
            return {**state, "response": "No research sections completed."}
        
        # Combine all section content for analysis
        sections_text = ""
        for section_id, section_data in completed_sections.items():
            sections_text += f"\n\n## {section_data['title']}\n{section_data['content']}"
        
        # Generate the complete report
        report_prompt = PromptTemplate.from_template("""
You are a Research Finalizer tasked with creating a comprehensive research report.

Topic: {topic}
Research Sections: {sections_content}

Create a complete research report with the following structure:

# {topic}: A Comprehensive Research Report

## Executive Summary
[Write a concise 2-3 paragraph executive summary that captures the key insights and main findings from all research sections]

## Key Findings
[List 5-7 key findings as bullet points, each with a brief explanation]

{detailed_sections}

## Limitations and Further Research
[Identify limitations of the current research and suggest 3-4 areas for further investigation]

---
*Research conducted using web search and analysis*

Format the report professionally and ensure all sections flow logically.
Make sure the Executive Summary and Key Findings accurately reflect the content of the detailed sections.
""")
        
        # Format detailed sections for inclusion
        detailed_sections_formatted = ""
        for section_id, section_data in completed_sections.items():
            detailed_sections_formatted += f"\n## {section_data['title']}\n{section_data['content']}\n"
        
        final_prompt = report_prompt.invoke({
            "topic": state["topic"],
            "sections_content": sections_text,
            "detailed_sections": detailed_sections_formatted
        })
        
        # Generate final report
        final_response = self.llm.invoke(final_prompt)
        final_report = final_response.content.strip()
        
        return {
            **state,
            "final_report": final_report,
            "response": final_report
        }
    
    def _build_graph(self) -> StateGraph:
        """Build the multi-agent workflow graph."""
        workflow = StateGraph(DeepResearchState)
        
        # Add nodes
        workflow.add_node("research_manager", self._research_manager)
        workflow.add_node("research_agent", self._specialized_research_agent)
        workflow.add_node("finalizer", self._finalizer)
        
        # Build the workflow
        workflow.add_edge(START, "research_manager")
        workflow.add_edge("research_manager", "research_agent")
        
        # Conditional edge to determine next step
        workflow.add_conditional_edges(
            "research_agent",
            self._should_continue_research,
            {
                "research_agent": "research_agent",  # Continue with next section
                "finalizer": "finalizer"              # All sections done, finalize
            }
        )
        
        workflow.add_edge("finalizer", END)
        
        return workflow.compile()
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a research query using the multi-agent system.
        
        Args:
            message: The research topic/query
            chat_history: Previous conversation history (not used in deep research)
            
        Returns:
            str: A comprehensive research report
        """
        if not self.graph:
            return "System not initialized. Please call initialize() first."
        
        # Initialize state for the research process
        initial_state = DeepResearchState(
            topic=message,
            research_plan=None,
            current_section_index=0,
            completed_sections={},
            final_report=None,
            response=""
        )
        
        try:
            # Run the multi-agent workflow
            result = self.graph.invoke(initial_state)
            
            # Return the final research report
            return result.get("response", "Research completed but no report generated.")
            
        except Exception as e:
            return f"An error occurred during research: {str(e)}"

