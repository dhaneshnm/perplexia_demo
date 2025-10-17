# Perplexia AI - Week 1: App Structure Documentation

## Overview

The Perplexia AI application is designed with a modular structure to support the staged development of an AI assistant. At its core, the design cleanly separates foundational logic (chat handling and tools) from implementation-specific features, allowing for progressive complexity and feature development across three parts: query understanding, basic tools, and memory management.

## Project Structure

```
perplexia_ai/
├── core/
│   ├── __init__.py
│   └── chat_interface.py          → Core chat interface definition
├── tools/
│   ├── __init__.py
│   └── calculator.py              → Calculator utility functions
├── week1/                         → STUDENT CODE (starter templates)
│   ├── __init__.py
│   ├── factory.py                 → Factory that returns the appropriate part implementation
│   ├── part1.py                   → Query understanding (STARTER CODE)
│   ├── part2.py                   → Basic tools (STARTER CODE)
│   └── part3.py                   → Memory and context handling (STARTER CODE)
├── solutions/                     → SOLUTION CODE
│   └── week1/
│       ├── __init__.py
│       ├── factory.py             → Factory for solutions
│       ├── part1.py               → Query understanding (COMPLETE SOLUTION)
│       ├── part2.py               → Basic tools (COMPLETE SOLUTION)
│       └── part3.py               → Memory (COMPLETE SOLUTION)
├── app.py                         → Main application logic and Gradio setup
└── __init__.py                    → Package initializer
```

## Student vs Solution Code

The project is organized to support learning and development:

- **`week1/` directory**: Contains starter code where students implement their solutions. These files have TODO comments and skeleton implementations to guide student development.

- **`solutions/week1/` directory**: Contains complete, working implementations. Students can reference these to understand the expected solution or run them to see the correct behavior.

- **Switching between them**: Use the `--solution` flag when running the application. Without the flag, your student code runs. With the flag, the solution code runs. This makes it easy to test your work and compare with the solution without moving files around.

## Core Concept: Chat Interface Design

### ChatInterface: The AI Assistant Blueprint

The `ChatInterface` is an abstract base class that all chat implementations inherit from. It defines a consistent contract with one key method:

- **`process_message(message, chat_history)`**: The core method that processes user input and returns a response based on current and past interactions. All initialization happens in `__init__()`.

## Implementation Parts

Built on top of the `ChatInterface`, the implementation focuses on fundamental AI assistant capabilities:

- **QueryUnderstandingChat (Part 1)**: Focuses on classifying and understanding different types of user queries
- **BasicToolsChat (Part 2)**: Introduces simple tool use (e.g., calculator integration)
- **MemoryChat (Part 3)**: Adds support for maintaining conversation context and memory

## System Flow Overview

1. **Application Startup**: The user starts the application by running `run.py` with `--week`, `--mode`, and optionally `--solution` arguments (e.g., `python run.py --week 1 --mode part2`)

2. **Code Selection**: Based on the `--solution` flag, the application imports from either `week1/` (student code) or `solutions/week1/` (solution code)

3. **Mode Resolution**: The specified mode is mapped to the appropriate enum value (`Week1Mode`) and passed to the factory method

4. **Chat Implementation Instantiation**: The factory method instantiates the appropriate `ChatInterface` implementation, with all component initialization (LLMs, tools, memory) happening in `__init__()`

5. **Web Interface Launch**: A Gradio web interface is launched, providing an interactive chat interface with appropriate examples and descriptions. The title indicates whether "Student" or "Solution" code is running.

6. **Message Processing Loop**: As the user sends messages, Gradio calls the `respond()` function, which delegates handling to the chat implementation's `process_message()` method

7. **Response Generation**: The chat implementation processes the message according to its specific logic and returns a response to be displayed in the interface

### Gradio Integration
The application uses Gradio for the web interface, providing:
- Automatic chat interface generation
- Built-in message history handling
- Easy deployment and sharing
- Customizable examples and descriptions per mode

## Running the Application

### Running Student Code (Default)

By default, the application runs your student implementations from `week1/`:

```bash
# Run Week 1, Part 1 (Query Understanding) - Student Code
python run.py --week 1 --mode part1

# Run Week 1, Part 2 (Basic Tools) - Student Code
python run.py --week 1 --mode part2  

# Run Week 1, Part 3 (Memory) - Student Code
python run.py --week 1 --mode part3
```

### Running Solution Code

To run the complete solution implementations from `solutions/week1/`, add the `--solution` flag:

```bash
# Run Week 1, Part 1 (Query Understanding) - Solution
python run.py --week 1 --mode part1 --solution

# Run Week 1, Part 2 (Basic Tools) - Solution
python run.py --week 1 --mode part2 --solution

# Run Week 1, Part 3 (Memory) - Solution
python run.py --week 1 --mode part3 --solution
```

### Development Workflow

1. **Write your code**: Implement the logic in `week1/part1.py`, `part2.py`, or `part3.py`
2. **Test your implementation**: Run `python run.py --week 1 --mode part1` to test your code
3. **Compare with solution**: Run `python run.py --week 1 --mode part1 --solution` to see the working implementation
4. **Iterate**: Switch between your code and solution as needed using the `--solution` flag

Each mode launches a tailored Gradio interface with appropriate examples and descriptions. The interface title will indicate whether you're running "Student" or "Solution" code.
