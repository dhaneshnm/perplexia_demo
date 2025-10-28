"""Part 2 - Agentic RAG implementation.

This implementation focuses on:
- Building an agent that can dynamically control its search strategy
- Combining document retrieval with web search
- Making autonomous decisions about information gathering
"""

import os
import chromadb
from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import create_retriever_tool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from pathlib import Path


class AgenticRAGChat(ChatInterface):
    """Week 3 Part 2 implementation focusing on Agentic RAG."""
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.retriever_tool = None
        self.web_search_tool = None
        self.graph = None
        self.document_paths = []
    
    def initialize(self) -> None:
        """Initialize components for Agentic RAG.
        
        Students should:
        - Initialize the chat model
        - Set up document vector store with OPM documents
        - Create tools for document retrieval and web search
        - Build an agent that can autonomously decide which tools to use
        """
        # Initialize OpenAI components
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
        self.llm = ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=0.0,
            api_key=openai_api_key,
        )
        
        # Initialize vector store with existing collection
        self.vector_store = Chroma(
            collection_name="week3_collection",
            embedding_function=self.embeddings,
            persist_directory="./chroma_langchain_db"
        )
        
        # Check if we need to load documents
        try:
            # Test if the collection has any documents
            test_results = self.vector_store.similarity_search("test", k=1)
            if not test_results:
                self._load_and_process_documents("./RAGDataset")
        except:
            self._load_and_process_documents("./RAGDataset")
        
        # Create retriever tool from vector store
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        self.retriever_tool = create_retriever_tool(
            retriever,
            "document_retrieval",
            "Search and retrieve information from OPM annual reports and organizational documents. "
            "Use this tool when users ask about OPM policies, annual reports, organizational information, "
            "federal employment data, or any government-related topics that might be in the document collection."
        )
        
        # Initialize Tavily web search tool
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
            
        self.web_search_tool = TavilySearch(
            tavily_api_key=tavily_api_key,
            max_results=3,
            topic="general",
            include_answer=True,
            include_raw_content=False
        )
        
        # Build the agent graph
        self.graph = self._build_graph()
    
    def _load_and_process_documents(self, dataset_path: str) -> None:
        """Load and process documents from the specified path."""
        path = Path(dataset_path)
        if not path.exists():
            print(f"Dataset directory not found at {path}")
            return
            
        # Create document loader for PDF files
        loader = DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        try:
            print("Loading PDF documents...")
            documents = loader.load()
            
            if not documents:
                print("No PDF documents found in the dataset directory")
                return
                
            print(f"Found {len(documents)} PDF documents")
            
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
                is_separator_regex=False,
            )
            
            # Split documents into chunks
            print("Processing documents into chunks...")
            splits = text_splitter.split_documents(documents)
            
            # Add documents to vector store
            if splits:
                print(f"Adding {len(splits)} chunks to vector store...")
                self.vector_store.add_documents(splits)
                print(f"Successfully added {len(splits)} document chunks to vector store")
                
                # Store document paths for reference
                self.document_paths = [str(Path(doc.metadata.get('source', '')).name) for doc in documents]
                
        except Exception as e:
            print(f"Error processing PDF documents: {str(e)}")
    
    def _build_graph(self):
        """Build the agentic RAG graph using create_react_agent."""
        # Combine all tools
        tools = [self.retriever_tool, self.web_search_tool]
        
        # Create the ReAct agent
        return create_react_agent(self.llm, tools)
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the Agentic RAG system.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response
        """
        if not self.graph:
            return "System not initialized. Please call initialize() first."
        
        # Create system message with tool usage instructions
        system_message = SystemMessage(content="""You are an intelligent assistant that can help users by searching through organizational documents and web resources.

Available tools:
- document_retrieval: Search and retrieve information from OPM annual reports and organizational documents. Use this for questions about OPM policies, federal employment, organizational data, annual reports, or government-related topics.
- tavily_search_results_json: Search the web for current information, news, facts, or topics not covered in the document collection.

Instructions:
- For questions about OPM, federal employment, organizational policies, annual reports, or government data, ALWAYS use the document_retrieval tool first.
- If the document search doesn't provide sufficient information, or if the question is about current events, recent developments, or general knowledge, use the web search tool.
- You can use both tools if needed to provide a comprehensive answer.
- Always cite your sources and be clear about whether information comes from the document collection or web search.
- If you use the document retrieval tool, include relevant citations and document references in your response.
- Provide detailed, informative answers that combine information from multiple sources when appropriate.""")
        
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
        
        try:
            # Invoke the ReAct agent
            result = self.graph.invoke({"messages": messages})
            
            # Extract the last AI message from the result
            if 'messages' in result and result['messages']:
                last_message = result['messages'][-1]
                if hasattr(last_message, 'content'):
                    return last_message.content.strip()
            
            return "I apologize, but I couldn't process your request."
            
        except Exception as e:
            return f"An error occurred while processing your request: {str(e)}"

