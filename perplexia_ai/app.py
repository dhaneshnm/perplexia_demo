import os
import gradio as gr
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_demo(week: int = 1, mode_str: str = "part1", use_solution: bool = False):
    """Create and return a Gradio demo with the specified mode.
    
    Args:
        week: Week implementation (only week 1 is supported)
        mode_str: String representation of the mode ('part1', 'part2', or 'part3')
        use_solution: If True, use solution implementation; if False, use student code
        
    Returns:
        gr.ChatInterface: Configured Gradio chat interface
    """
    if week != 1:
        raise ValueError(f"Week {week} is not supported. Only week 1 is supported.")
    
    # Import the appropriate factory based on use_solution flag
    if use_solution:
        from perplexia_ai.solutions.week1.factory import Week1Mode, create_chat_implementation as create_week1_chat
        code_type = "Solution"
    else:
        from perplexia_ai.week1.factory import Week1Mode, create_chat_implementation as create_week1_chat
        code_type = "Student"

    # Convert string to enum
    mode_map = {
        "part1": Week1Mode.PART1_QUERY_UNDERSTANDING,
        "part2": Week1Mode.PART2_BASIC_TOOLS,
        "part3": Week1Mode.PART3_MEMORY
    }
    
    if mode_str not in mode_map:
        raise ValueError(f"Unknown mode: {mode_str}. Choose from: {list(mode_map.keys())}")
    
    mode = mode_map[mode_str]
    
    # Create the chat implementation (initialization happens in __init__)
    chat_interface = create_week1_chat(mode)
    
    # Create the Gradio interface with appropriate title based on mode
    titles = {
        "part1": f"Perplexia AI - Week 1: Query Understanding ({code_type})",
        "part2": f"Perplexia AI - Week 1: Basic Tools ({code_type})",
        "part3": f"Perplexia AI - Week 1: Memory ({code_type})"
    }
    
    descriptions = {
        "part1": "Your intelligent AI assistant that can understand different types of questions and format responses accordingly.",
        "part2": "Your intelligent AI assistant that can answer questions, perform calculations, and format responses.",
        "part3": "Your intelligent AI assistant that can answer questions, perform calculations, and maintain conversation context."
    }
    
    # Create the respond function that uses our chat implementation
    def respond(message: str, history: List[Tuple[str, str]]) -> str:
        """Process the message and return a response.
        
        Args:
            message: The user's input message
            history: List of previous (user, assistant) message tuples
            
        Returns:
            str: The assistant's response
        """
        # Get response from our chat implementation
        return chat_interface.process_message(message, history)
    
    # Create example queries for Week 1
    examples = [
        ["What is machine learning?"],
        ["Compare SQL and NoSQL databases"],
        ["If I have a dinner bill of $120, what would be a 15% tip?"],
        ["What about 20%?"],
    ]
    
    # Create the Gradio interface
    demo = gr.ChatInterface(
        fn=respond,
        title=titles[mode_str],
        type="messages",
        description=descriptions[mode_str],
        examples=examples,
        theme=gr.themes.Soft()
    )
    
    return demo
