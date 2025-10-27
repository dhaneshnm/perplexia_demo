"""Part 3 - Combined Web and Document Search implementation using LangGraph.

This implementation combines the capabilities of Part 1 (Web Search) and Part 2 (Document RAG)
to provide a comprehensive search and response system that leverages both web content
and local documents.

Key features:
- Unified search across web and document sources
- Intelligent source selection and result combination
- Comprehensive response generation with citations from both sources
"""

import os
import json
import chromadb
from typing import Dict, List, Optional, TypedDict, Any, Tuple
from pathlib import Path

from perplexia_ai.core.chat_interface import ChatInterface
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import List, Sequence


class State(TypedDict):
    """Combined state for web and document search workflow."""
    question: str
    response: str
    chat_history: Optional[List[Dict[str, str]]]
    # Web search specific
    web_search_results: Optional[str]
    search_iteration: int
    accumulated_web_results: List[Any]
    # Document search specific
    retrieved_documents: Optional[List[Dict[str, Any]]]
    document_context: Optional[List[Dict[str, Any]]]


class CorrectiveRAGChat(ChatInterface):
    """Week 2 Part 3 implementation combining web search and document RAG."""
    
    def __init__(self):
        # Core components
        self.llm = None
        self.graph = None
        
        # Web search components
        self.search_tool = None
        
        # Document search components
        self.embeddings = None
        self.vector_store = None
        self.document_paths = []
        
    def initialize(self) -> None:
        """Initialize components for both web search and document RAG.
        
        This method:
        1. Sets up OpenAI components (LLM and embeddings)
        2. Initializes Tavily web search
        3. Sets up document processing and vector store
        4. Creates the combined workflow graph
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
            api_key=openai_api_key
        )
        
        # Set up Tavily web search
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError(
                "TAVILY_API_KEY environment variable is required for web search"
            )
        
        self.search_tool = TavilySearch(
            tavily_api_key=tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=True
        )
        self.llm.bind_tools([self.search_tool])
        
        # Initialize vector store with a clean collection
        client = chromadb.PersistentClient(path="./chroma_langchain_db")
        
        # Delete the collection if it exists
        try:
            client.delete_collection(name="week2_collection")
        except ValueError:
            # Collection doesn't exist, which is fine
            pass
            
        # Create a fresh collection
        self.vector_store = Chroma(
            collection_name="week2_collection",
            embedding_function=self.embeddings,
            persist_directory="./chroma_langchain_db"
        )
        
        # Process and load documents
        self._load_and_process_documents("./RAGDataset")
        
        # Build the combined workflow graph
        self.graph = self._build_graph()
        
    def _load_and_process_documents(self, dataset_path: str) -> None:
        """Load and process documents from the specified path.
        
        Args:
            dataset_path: Path to the directory containing PDF documents
            
        Raises:
            ValueError: If the dataset directory doesn't exist
            Exception: If there's an error processing the documents
        """
        path = Path(dataset_path)
        if not path.exists():
            raise ValueError(f"Dataset directory not found at {path}")
            
        # Create document loader for PDF files in the dataset directory
        loader = DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        try:
            # Load documents
            print("Loading PDF documents...")
            documents = loader.load()
            
            if not documents:
                print("No PDF documents found in the dataset directory")
                return
                
            print(f"Found {len(documents)} PDF documents")
            
            # Create text splitter for chunking
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
            
            # Add documents to vector store with embeddings
            if splits:
                print(f"Adding {len(splits)} chunks to vector store...")
                self.vector_store.add_documents(splits)
                print(f"Successfully added {len(splits)} document chunks to vector store")
                
                # Store document paths for citation
                self.document_paths = [str(Path(doc.metadata.get('source', '')).name) for doc in documents]
                
        except Exception as e:
            print(f"Error processing PDF documents: {str(e)}")
            raise
            
    def _call_web_search(self, state: State) -> State:
        """Invoke the TavilySearch tool and attach results to the state."""
        if not self.search_tool:
            return {**state, "response": "Web search tool is not available.", "web_search_results": None}
            
        try:
            raw = self.search_tool.invoke({"query": state["question"]})
        except Exception as e:
            return {**state, "response": f"Web search failed: {e}", "web_search_results": None}
            
        # Accumulate web results
        accumulated = state.get("accumulated_web_results", [])
        accumulated.append(raw)
        
        # Update iteration counter
        iteration = state.get("search_iteration", 0) + 1
        
        return {
            **state,
            "web_search_results": raw,
            "search_iteration": iteration,
            "accumulated_web_results": accumulated
        }
        
    def _retrieve_documents(self, state: State) -> State:
        """Retrieve relevant documents from the vector store."""
        if not self.vector_store:
            return {**state, "response": "Vector store not initialized", "retrieved_documents": None}
            
        try:
            # Search for relevant documents
            results = self.vector_store.similarity_search_with_score(
                state["question"],
                k=5  # Get top 5 most relevant documents
            )
            
            # Format retrieved documents
            retrieved_docs = []
            for doc, score in results:
                retrieved_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": score,
                    "source": str(Path(doc.metadata.get('source', '')).name)
                })
                
            return {**state, "retrieved_documents": retrieved_docs}
            
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return {**state, "retrieved_documents": None}
            
    def _format_web_results(self, results: List[Any]) -> str:
        """Format web search results with citations for summary generation."""
        formatted_results = []
        for i, search_result in enumerate(results, 1):
            if isinstance(search_result, dict) and "results" in search_result:
                for j, result in enumerate(search_result["results"], 1):
                    formatted_results.append({
                        "citation": f"[W{i}.{j}]",  # W prefix for web sources
                        "url": result.get('url', 'No URL'),
                        "content": result.get('content', ''),
                        "title": result.get('title', ''),
                        "published_date": result.get('published_date', '')
                    })
        return json.dumps({"sources": formatted_results})
        
    def _format_doc_results(self, docs: List[Dict[str, Any]]) -> str:
        """Format document search results with citations for summary generation."""
        formatted_results = []
        for i, doc in enumerate(docs, 1):
            formatted_results.append({
                "citation": f"[D{i}]",  # D prefix for document sources
                "source": doc["source"],
                "content": doc["content"],
                "relevance_score": doc["relevance_score"]
            })
        return json.dumps({"sources": formatted_results})
        
    def _generate_combined_response(self, state: State) -> State:
        """Generate a response using both web and document search results."""
        # Check if we have any results to work with
        if not state.get("retrieved_documents") and not state.get("accumulated_web_results"):
            return {**state, "response": "No relevant information found from either source."}
            
        # Format all available sources
        context_parts = []
        
        # Add document results if available
        if state.get("retrieved_documents"):
            doc_data = json.loads(self._format_doc_results(state["retrieved_documents"]))
            context_parts.append("Document Sources:")
            for source in doc_data["sources"]:
                context_parts.append(
                    f"{source['citation']} (from {source['source']}):\\n{source['content']}"
                )
                
        # Add web results if available
        if state.get("accumulated_web_results"):
            web_data = json.loads(self._format_web_results(state["accumulated_web_results"]))
            context_parts.append("\\nWeb Sources:")
            for source in web_data["sources"]:
                context_parts.append(
                    f"{source['citation']} ({source['title']}):\\n{source['content']}"
                )
                
        context_str = "\\n\\n".join(context_parts)
        
        prompt = PromptTemplate.from_template(
            "You are a knowledgeable assistant that combines information from both local documents "
            "and web sources. Use the provided information to answer the user's question comprehensively.\n\n"
            "Important Citation Rules:\n"
            "- Use [D1], [D2], etc. when citing from documents\n"
            "- Use [W1.1], [W1.2], etc. when citing from web sources\n"
            "- Cite multiple sources to support each claim when possible\n"
            "- Clearly indicate when information comes from different sources\n"
            "- If sources conflict, acknowledge the discrepancy\n\n"
            "Available Information:\n{context}\n\n"
            "Question: {question}\n\n"
            "Please provide a detailed answer with appropriate citations:"
        ).invoke({
            "context": context_str,
            "question": state["question"]
        })
        
        response = self.llm.invoke(prompt)
        return {**state, "response": response.content}
        
    def _build_graph(self) -> StateGraph:
        """Build the combined search workflow graph."""
        workflow = StateGraph(State)
        
        # Add nodes for each major operation
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("generate_response", self._generate_combined_response)
        
        # Build sequential workflow
        workflow.add_edge(START, "process_input")
        workflow.add_edge("process_input", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
        
    def _process_input(self, state: State) -> State:
        """Initial processing step that performs web search."""
        # First web search
        try:
            if not self.search_tool:
                return {**state, "response": "Web search tool is not available."}
                
            raw = self.search_tool.invoke({"query": state["question"]})
            accumulated = [raw]
            
            # Second web search iteration
            raw2 = self.search_tool.invoke({"query": state["question"]})
            accumulated.append(raw2)
            
            return {
                **state,
                "web_search_results": raw2,
                "search_iteration": 2,
                "accumulated_web_results": accumulated
            }
            
        except Exception as e:
            return {**state, "response": f"Web search failed: {e}"}
        
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using both web search and document RAG.
        
        This method:
        1. Performs parallel document and web searches
        2. Combines results from both sources
        3. Generates a comprehensive response with citations
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response combining both web and document knowledge
        """
        if not self.graph:
            return "System not initialized. Please call initialize() first."
            
        initial_state = State(
            question=message,
            chat_history=chat_history,
            response="",
            web_search_results=None,
            search_iteration=0,
            accumulated_web_results=[],
            retrieved_documents=None,
            document_context=None
        )
        
        try:
            result = self.graph.invoke(initial_state)
            response = result.get('response', '')
            return response.strip()
        except Exception as e:
            return f"Error processing message: {str(e)}"


