import os 
import json
import requests
# from anthropic import Anthropic

# from dotenv import load_dotenv

# load_dotenv()

# client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Ollama API endpoint (default local installation)
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:8b"

# ----------TOOL DEFINITIONS----------

# # This is what we tell the Claude about our tools
# TOOLS = [
#     {
#         "name": "list_directory",
#         "description": "List all files and folders in a directory. Use this to explore the project structure.",
#         "input_schema": {
#             "type": "object", # Top level is object
#             "properties": {
#                 "path": {
#                     "type": "string",
#                     "description": "The directory path to list. Use '.' for current directory.",
#                 }
#             },
#             "required": ["path"],
#         },
#     },
#     {
#         "name": "read_file",
#         "description": "Read the contents of a file. Use this to understand what code does.",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "path": {
#                     "type": "string",
#                     "description": "The file path to read."
#                 }
#             },
#             "required": ["path"],
#         },
#     },
#     {
#         "name": "write_file",
#         "description": "Write content to a file. Use this to save your analysis.",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "path": {
#                     "type": "string",
#                     "description": "The file path to write to."
#                 },
#                 "content": {
#                     "type": "string",
#                     "description": "The content to write."
#                 },
#             },
#             "required": ["path", "content"],
#         },
#     },
# ]

# Ollama expects tools in OpenAI format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List all files and folders in a directory. Use this to explore the project structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list. Use '.' for current directory.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Use this to understand what code does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to read."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Use this to save your analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to write to."},
                    "content": {"type": "string", "description": "The content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
]

# ----------TOOL IMPLEMENTATIONS----------

def list_directory(path: str) -> str:
    """List contents of a directory."""
    try:
        items = os.listdir(path)
        # Separate files and directories
        dirs = [
            f"📁 {item}/" for item in items if os.path.isdir(os.path.join(path, item))
        ]
        files = [
            f"📄 {item}" for item in items if os.path.isfile(os.path.join(path, item))
        ]

        result = f"Contents of {path}:\n"
        result += "\n".join(sorted(dirs) + sorted(files))
        return result if items else "Directory is empty."
    except Exception as e:
        return f"Error listing directory: {str(e)}"


def read_file(path: str) -> str:
    """Read contents of a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Truncate very long files to avoid token limits
        if len(content) > 10000:
            content = (
                content[:10000] + "\n\n... [FILE TRUNCATED - Too long to display fully]"
            )

        return f"Contents of '{path}':\n\n```\n{content}\n```"
    except FileNotFoundError:
        return f"Error: File '{path}' not found."
    except PermissionError:
        return f"Error: Permission denied for '{path}'."
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a text file (binary content)."
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        # Create directories if they don't exist
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"✅ Successfully wrote {len(content)} characters to '{path}'"
    except PermissionError:
        return f"Error: Permission denied for '{path}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"
    

# Map tool names to functions
TOOL_FUNCTIONS = {
    "list_directory": list_directory,
    "read_file": read_file,
    "write_file": write_file,
}

# ----------AGENT LOGIC----------

# THE AGENT LOOP

def run_agent(task: str, max_iterations: int = 15) -> str:
    """
    Run the agent loop.

    The core pattern of ALL AI agents:
    1. THINK  - LLM decides what to do next
    2. ACT    - Execute the chosen tool
    3. OBSERVE - See the result
    4. REPEAT - Until task is complete

    Args:
        task: The task for the agent to complete
        max_iterations: Safety limit to prevent infinite loops

    Returns:
        The agent's final response
    """

    print("\n" + "=" * 60)
    print("🤖 AI AGENT STARTING")
    print("=" * 60)
    print(f"\n📋 TASK: {task}\n")
    print("-" * 60)

    ## Claude
    # Initialize conversation with the user's task
    # messages = [{"role": "user", "content": task}]

    # System prompt - shapes agent behavior and personality
    system_prompt = """You are a Project Analyzer agent - an AI that explores codebases and writes clear documentation.

## Your Approach

When given a project to analyze:

1. **EXPLORE** - First, list the root directory to understand the project structure
2. **INVESTIGATE** - Read key files: README, main entry points, configuration files
3. **UNDERSTAND** - Identify the purpose, technologies, and architecture
4. **DOCUMENT** - Write a comprehensive ANALYSIS.md summarizing your findings

## Guidelines

- Be thorough but efficient - don't read every file, focus on important ones
- Skip binary files, node_modules, __pycache__, .git directories
- Look for patterns: package.json, requirements.txt, Cargo.toml tell you the tech stack
- Entry points are usually: main.py, index.js, app.py, src/main.rs, etc.
- Configuration files reveal a lot: .env.example, config files, docker files

## Output Format

Your ANALYSIS.md should include:
- Project Overview (what it does)
- Tech Stack (languages, frameworks, key dependencies)
- Project Structure (key directories and their purposes)
- Key Files (most important files and what they do)
- How to Run (if you can determine this)
- Notes (anything interesting you discovered)

When you're finished writing the analysis, provide a brief summary to the user."""

    # Initialize conversation with system prompt and user task
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task}
    ]

    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n🔄 ITERATION {iteration}/{max_iterations}")
        print("-" * 40)

        # ========================================
        # STEP 1: THINK
        # Ask Claude what to do next
        # ========================================

        ## Claude
        # response = client.messages.create(
        #     model="claude-sonnet-4-20250514",
        #     max_tokens=4096,
        #     system=system_prompt,
        #     tools=TOOLS,
        #     messages=messages,
        # )

        # Ollama
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
        }

        try:
            response = requests.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            # print(result)
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            break

        message = result.get("message", {})

        # ========================================
        # STEP 2 & 3: ACT & OBSERVE
        # Execute tools and collect results
        # ========================================

        ## Claude
        # if response.stop_reason == "tool_use":
        #     # Claude wants to use one or more tools
        #     tool_results = []

        #     for block in response.content:
        #         if block.type == "tool_use":
        #             tool_name = block.name
        #             tool_input = block.input
        #             tool_use_id = block.id

        #             # Display what the agent is doing
        #             print(f"\n🔧 TOOL CALL: {tool_name}")
        #             for key, value in tool_input.items():
        #                 display_value = (
        #                     value if len(str(value)) < 50 else str(value)[:50] + "..."
        #                 )
        #                 print(f"   └─ {key}: {display_value}")

        #             # Execute the tool
        #             if tool_name in TOOL_FUNCTIONS:
        #                 result = TOOL_FUNCTIONS[tool_name](**tool_input)
        #             else:
        #                 result = f"Error: Unknown tool '{tool_name}'"

        #             # Show abbreviated result
        #             result_preview = (
        #                 result[:200] + "..." if len(result) > 200 else result
        #             )
        #             print(f"\n   📤 RESULT: {result_preview}")

        #             # Collect tool result for Claude
        #             tool_results.append(
        #                 {
        #                     "type": "tool_result",
        #                     "tool_use_id": tool_use_id,
        #                     "content": result,
        #                 }
        #             )

        #         elif block.type == "text" and block.text.strip():
        #             # Claude is thinking out loud
        #             print(f"\n💭 THINKING: {block.text[:200]}...")

        #     # Add this exchange to conversation history
        #     messages.append({"role": "assistant", "content": response.content})
        #     messages.append({"role": "user", "content": tool_results})

        ## Ollama
        # Check if there are tool calls
        if message.get("tool_calls"):
            tool_results = []

            for tool_call in message["tool_calls"]:
                function = tool_call.get("function", {})
                tool_name = function.get("name")
                tool_args = function.get("arguments", {})

                # Display what the agent is doing
                print(f"\n🔧 TOOL CALL: {tool_name}")
                for key, value in tool_args.items():
                    display_value = (
                        value if len(str(value)) < 50 else str(value)[:50] + "..."
                    )
                    print(f"   └─ {key}: {display_value}")

                # Execute the tool
                if tool_name in TOOL_FUNCTIONS:
                    result_content = TOOL_FUNCTIONS[tool_name](**tool_args)
                else:
                    result_content = f"Error: Unknown tool '{tool_name}'"

                # Show abbreviated result
                result_preview = (
                    result_content[:200] + "..." if len(result_content) > 200 else result_content
                )
                print(f"\n   📤 RESULT: {result_preview}")

                # Collect tool result
                tool_results.append({
                    "role": "tool",
                    "content": result_content,
                })

            # Add assistant message with tool calls to conversation
            messages.append(message)
            
            # Add tool results to conversation
            messages.extend(tool_results)
        else:
            # ========================================
            # COMPLETE
            # Model finished - no more tool calls
            # ========================================

            ## Claude
            # final_response = ""
            # for block in response.content:
            #     if hasattr(block, "text"):
            #         final_response += block.text

            # Ollama
            final_response = message.get("content", "")

            # Check if there's thinking text
            if final_response:
                print(f"\n💭 RESPONSE: {final_response[:200]}...")

            print("\n" + "=" * 60)
            print("✅ AGENT COMPLETE")
            print("=" * 60)
            print(f"\n{final_response}\n")

            return final_response

    # Safety: hit max iterations
    print("\n" + "=" * 60)
    print("⚠️  MAX ITERATIONS REACHED")
    print("=" * 60)
    return "Agent stopped: Maximum iterations reached. Task may be incomplete."


# ============================================================
# MAIN - Run the agent
# ============================================================

if __name__ == "__main__":
    import sys

    # Get project path from command line or use current directory
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
    else:
        project_path = "."

    # Make sure path exists
    if not os.path.exists(project_path):
        print(f"Error: Path '{project_path}' does not exist.")
        sys.exit(1)

    # Define the task
    task = f"""Analyze the project located at '{project_path}'.

Your mission:
1. Explore the project structure
2. Read and understand the key files
3. Write a comprehensive ANALYSIS.md file in the project root

Focus on helping a new developer understand this codebase quickly."""

    # Run the agent
    result = run_agent(task)