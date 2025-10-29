"""Part 3 - Deep Research Multi-Agent System implementation.

This implementation focuses on:
- Creating specialized agents for different research tasks
- Coordinating multiple agents for comprehensive research
- Generating structured research reports
- Managing complex multi-agent workflows
"""

import os
import json
from typing import Dict, List, Optional, Any
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


class DeepResearchChat(ChatInterface):
    """Week 3 Part 3 implementation focusing on multi-agent deep research."""
    
    def __init__(self):
        self.llm = None
        self.search_tool = None
        self.graph = None
        self.research_state = {}
    
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
            model='gpt-5-mini',
            #temperature=0.1,  # Lower temperature for more consistent research output
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
    
    def _create_research_planning_tool(self):
        """Create the research planning tool."""
        @tool("research_planning")
        def research_planning_tool(topic: str) -> str:
            """Create a comprehensive research plan for a given topic.
            
            Args:
                topic: The research topic to create a plan for
                
            Returns:
                str: A JSON string containing the research plan with sections
            """
            # Do initial web search to understand the topic scope
            try:
                initial_search = self.search_tool.invoke({"query": topic})
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
                "topic": topic,
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
                
                # Store the plan in research state
                self.research_state[topic] = {
                    "research_plan": research_plan,
                    "current_section_index": 0,
                    "completed_sections": {}
                }
                
                return json.dumps(research_plan, indent=2)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing research plan JSON: {e}")
                # Fallback plan
                research_plan = {
                    "sections": [
                        {
                            "section_id": "section_1",
                            "title": f"Overview of {topic}",
                            "research_prompt": f"Provide a comprehensive overview of {topic}",
                            "search_queries": [topic, f"{topic} overview", f"{topic} definition"]
                        }
                    ]
                }
                
                # Store the fallback plan
                self.research_state[topic] = {
                    "research_plan": research_plan,
                    "current_section_index": 0,
                    "completed_sections": {}
                }
                
                return json.dumps(research_plan, indent=2)
        
        return research_planning_tool
    
    def _create_specialized_research_tool(self):
        """Create the specialized research tool."""
        @tool("conduct_section_research")
        def conduct_section_research(topic: str, section_id: str) -> str:
            """Conduct detailed research for a specific section of the research plan.
            
            Args:
                topic: The main research topic
                section_id: The ID of the section to research (e.g., "section_1")
                
            Returns:
                str: The completed research content for the section
            """
            if topic not in self.research_state:
                return "No research plan found for this topic. Please create a research plan first."
            
            state = self.research_state[topic]
            research_plan = state.get("research_plan")
            
            if not research_plan or not research_plan.get("sections"):
                return "No research plan sections available"
            
            # Find the section by ID
            current_section = None
            for section in research_plan["sections"]:
                if section["section_id"] == section_id:
                    current_section = section
                    break
            
            if not current_section:
                return f"Section {section_id} not found in research plan"
            
            research_prompt = current_section["research_prompt"]
            search_queries = current_section.get("search_queries", [topic])
            
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
6. Make the section comprehensive but focused.
7. Make sure that the section has atleast 250 words of content.
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
            
            # Store completed section
            state["completed_sections"][section_id] = {
                "title": current_section["title"],
                "content": section_content
            }
            
            return f"Section research completed for '{current_section['title']}':\n\n{section_content}"
        
        return conduct_section_research
    
    def _create_report_finalizer_tool(self):
        """Create the report finalizer tool."""
        @tool("finalize_research_report")
        def finalize_research_report(topic: str) -> str:
            """Generate the final comprehensive research report from all completed sections.
            
            Args:
                topic: The main research topic
                
            Returns:
                str: The complete formatted research report
            """
            if topic not in self.research_state:
                return "No research data found for this topic."
            
            state = self.research_state[topic]
            completed_sections = state.get("completed_sections", {})
            
            if not completed_sections:
                return "No research sections completed."
            
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
                "topic": topic,
                "sections_content": sections_text,
                "detailed_sections": detailed_sections_formatted
            })
            
            # Generate final report
            final_response = self.llm.invoke(final_prompt)
            final_report = final_response.content.strip()
            
            # Store the final report
            state["final_report"] = final_report
            
            return final_report
        
        return finalize_research_report

    
    def _build_graph(self):
        """Build the multi-agent research system using create_react_agent."""
        # Create all the tools
        research_planning_tool = self._create_research_planning_tool()
        specialized_research_tool = self._create_specialized_research_tool()
        report_finalizer_tool = self._create_report_finalizer_tool()
        
        # Combine tools with the web search tool
        tools = [research_planning_tool, specialized_research_tool, report_finalizer_tool, self.search_tool]
        
        # Create the ReAct agent
        return create_react_agent(self.llm, tools)
    
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
        
        # Create system message with comprehensive instructions
        system_message = SystemMessage(content="""You are an intelligent multi-agent research system that conducts comprehensive research on any topic. You have access to multiple specialized tools to perform deep research.

Available tools:
- research_planning: Create a structured research plan with multiple sections for a given topic
- conduct_section_research: Conduct detailed research for specific sections of the research plan  
- finalize_research_report: Generate a comprehensive final report from all completed research sections
- tavily_search_results_json: Search the web for additional information when needed

Research Process:
1. ALWAYS start by using the research_planning tool to create a comprehensive research plan for the topic
2. For each section in the research plan, use conduct_section_research with the section_id to research that specific area
3. Continue researching ALL sections in the plan systematically
4. Once all sections are completed, use finalize_research_report to create the final comprehensive report
5. You may use tavily_search_results_json for additional web searches if needed during the process

Instructions:
- Be thorough and systematic in your research approach
- Ensure all sections of the research plan are completed before finalizing the report
- The final report should be comprehensive, well-structured, and professionally formatted
- Include proper citations and references throughout the research
- Focus on providing in-depth analysis and insights, not just surface-level information

Remember: This is a multi-step process that requires using multiple tools in sequence to produce a comprehensive research report.""")
        
        # Start with system message
        messages = [system_message]
        
        # Convert chat history to message format if provided
        if chat_history:
            for entry in chat_history:
                if 'user' in entry and 'assistant' in entry:
                    messages.append(HumanMessage(content=entry['user']))
                    messages.append(AIMessage(content=entry['assistant']))
        
        # Add the current user message
        messages.append(HumanMessage(content=f"Please conduct comprehensive research on: {message}"))
        
        try:
            # Invoke the ReAct agent
            result = self.graph.invoke({"messages": messages})
            
            # Extract the last three AI messages from the result
            if 'messages' in result and result['messages']:
                messages_list = result['messages']
                # Get the last 3 messages, or all messages if fewer than 3
                last_messages = messages_list[-3:] if len(messages_list) >= 3 else messages_list
                
                # Extract content from messages and join them
                content_parts = []
                for message in last_messages:
                    if hasattr(message, 'content') and message.content:
                        content_parts.append(message.content.strip())
                
                if content_parts:
                    return '\n\n---\n\n'.join(content_parts)
            
            return "Research completed but no response generated."
            
        except Exception as e:
            return f"An error occurred during research: {str(e)}"
    
