"""Part 2 - Document RAG implementation using LangGraph.

This implementatio        # Process and load documents
        self._load_and_process_documents("./RAGDataset")
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()focuses on:
- Setting up document loading and processing
- Creating vector embeddings and storage
- Implementing retrieval-augmented generation
- Formatting responses with citations from OPM documents
"""
import os
import chromadb
from typing import Dict, List, Optional, TypedDict, Any, Tuple
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from pathlib import Path

class State(TypedDict):
    """State for the RAG workflow."""
    question: str
    response: str
    chat_history: Optional[List[Dict[str, str]]]
    context: Optional[List[Dict[str, Any]]]
    retrieved_documents: Optional[List[Dict[str, Any]]]

class DocumentRAGChat(ChatInterface):
    """Week 2 Part 2 implementation for document RAG."""
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.document_paths = []
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for document RAG.
        
        Students should:
        - Initialize the LLM
        - Set up document loading and processing (e.g., OPM annual reports)
        - Create vector embeddings
        - Build retrieval system
        - Create LangGraph for RAG workflow
        """
        # Initialize OpenAI components
        opnai_api_key = os.getenv("OPENAI_API_KEY")
        if not opnai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=opnai_api_key)
        self.llm = ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=0.0,
            api_key=opnai_api_key,
            # callbacks=[self.tracer]
        )
        
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

        self.graph = self._build_graph()
        # Process and load documents
        self._load_and_process_documents("./RAGDataset")

    
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
            glob="**/*.pdf",  # Load all PDF files recursively
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
            # Using smaller chunks for PDF content to maintain context
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Smaller chunks for PDFs
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],  # PDF-specific separators
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
    
    def _generate_response(self, state: State) -> State:
        """Generate a response using retrieved documents."""
        if not state.get("retrieved_documents"):
            return {**state, "response": "No relevant documents found"}
        
        # Format documents for prompt
        context_str = ""
        for i, doc in enumerate(state["retrieved_documents"], 1):
            context_str += f"Source [{i}] {doc['source']}:\\n{doc['content']}\\n\\n"
        
        prompt = PromptTemplate.from_template(
            "You are a knowledgeable assistant helping users understand documents. "
            "Use the provided document excerpts to answer the user's question. "
            "Include relevant citations using the source numbers in square brackets (e.g. [1], [2]). "
            "If the documents don't contain enough information to answer fully, say so. "
            "Always cite your sources and be factual based on the provided context.\n\n"
            "Context Documents:\n{context}\n\n"
            "Question: {question}\n\n"
            "Please provide a detailed answer with citations:"
        ).invoke({
            "context": context_str,
            "question": state["question"]
        })
        
        response = self.llm.invoke(prompt)
        return {**state, "response": response.content}
    
    def _build_graph(self) -> StateGraph:
        """Build the RAG workflow graph."""
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_response)
        
        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using document RAG.
        
        Retrieves relevant documents and generates a response using LangGraph workflow.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response based on document knowledge
        """
        if not self.graph:
            return "System not initialized. Please call initialize() first."
            
        initial_state = State(
            question=message,
            chat_history=chat_history,
            response="",
            context=None,
            retrieved_documents=None
        )
        
        result = self.graph.invoke(initial_state)
        response = result.get('response', '')
        return response.strip()

