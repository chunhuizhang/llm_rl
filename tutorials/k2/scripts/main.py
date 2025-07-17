import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from openai import OpenAI
import requests
import math
import os
from pathlib import Path
from dotenv import load_dotenv
from tavily import TavilyClient
import argparse


@dataclass
class ToolCall:
    """Tool call representation"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Tool result representation"""
    tool_call_id: str
    content: str
    success: bool = True
    error: Optional[str] = None


class AgentTools:
    """Collection of available tools for the agent"""
    
    def __init__(self):
        self.tools_registry = {
            "search_web": self.search_web,
            "calculate": self.calculate,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "mark_task_complete": self.mark_task_complete
        }
        
        # Initialize Tavily client
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool definitions"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for real-time information using Tavily API. Returns structured results with titles, URLs, and content snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find information about"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read content from a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_task_complete",
                    "description": "Mark a task as complete",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "Summary of completed task"
                            }
                        },
                        "required": ["summary"]
                    }
                }
            }
        ]
    
    def search_web(self, query: str) -> str:
        """Search the web for information using Tavily API"""
        if self.tavily_client:
            try:
                # Use real Tavily API
                print(f"🔍 Searching web with Tavily API: {query}")
                response = self.tavily_client.search(
                    query=query,
                    search_depth="basic",
                    include_images=False,
                    include_answer=True,
                    max_results=5
                )
                
                # Format the search results
                formatted_results = []
                
                # Add answer if available
                if response.get("answer"):
                    formatted_results.append(f"📝 Answer: {response['answer']}")
                
                # Add search results
                if response.get("results"):
                    formatted_results.append("\n🔗 Search Results:")
                    for i, result in enumerate(response["results"], 1):
                        title = result.get("title", "No title")
                        url = result.get("url", "No URL")
                        content = result.get("content", "No content")
                        
                        # Truncate content if too long
                        if len(content) > 200:
                            content = content[:200] + "..."
                        
                        formatted_results.append(f"{i}. {title}")
                        formatted_results.append(f"   URL: {url}")
                        formatted_results.append(f"   Content: {content}")
                        formatted_results.append("")
                
                return "\n".join(formatted_results) if formatted_results else f"No results found for '{query}'"
                
            except Exception as e:
                print(f"❌ Tavily API error: {e}")
                return f"Search API error: {str(e)}"
        else:
            # Fallback to simulation mode
            print(f"🔧 Using simulation mode for search: {query}")
            time.sleep(1)  # Simulate network delay
            if "pietro schirano" in query.lower():
                return "Pietro Schirano is a designer and creative director known for his work in AI and design. He's active on social media and has worked with various tech companies."
            else:
                return f"Search results for '{query}': [Simulated search results]"
    
    def calculate(self, expression: str) -> str:
        """Perform mathematical calculations"""
        try:
            # Clean up expression and evaluate safely
            cleaned = expression.replace('x', '*').replace('÷', '/')
            result = eval(cleaned)
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"
    
    def read_file(self, file_path: str) -> str:
        """Read content from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"File content:\n{content}"
        except FileNotFoundError:
            return f"File not found: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def write_file(self, file_path: str, content: str) -> str:
        """Write content to a file"""
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def mark_task_complete(self, summary: str) -> str:
        """Mark a task as complete"""
        return f"Task completed: {summary}"


class OpenAIAgent:
    """OpenAI-style agent with tool calling and agentic loops"""
    
    def __init__(self, model_id: str = "gpt-4o-mini"):
        self.model_id = model_id
        self.api_key = os.getenv("OPENAI_API_KEY")
        if 'k2' in self.model_id:
            self.base_url = "https://api.moonshot.cn/v1"
            self.api_key = os.getenv("MOONSHOT_API_KEY")
        else:
            self.base_url = "https://api.openai.com/v1"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.tools = AgentTools()
        self.conversation_history = []
        self.max_iterations = 10
        self.current_iteration = 0
    
    def _execute_tool_calls_parallel(self, tool_calls: List[Dict]) -> List[ToolResult]:
        """Execute multiple tool calls in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tool calls to executor
            future_to_call = {}
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
                if function_name in self.tools.tools_registry:
                    future = executor.submit(
                        self.tools.tools_registry[function_name], 
                        **arguments
                    )
                    future_to_call[future] = tool_call
            
            # Collect results as they complete
            for future in as_completed(future_to_call):
                tool_call = future_to_call[future]
                try:
                    result = future.result()
                    results.append(ToolResult(
                        tool_call_id=tool_call["id"],
                        content=result,
                        success=True
                    ))
                except Exception as e:
                    results.append(ToolResult(
                        tool_call_id=tool_call["id"],
                        content="",
                        success=False,
                        error=str(e)
                    ))
        
        return results
    
    def _should_stop(self, response: Dict, tool_results: List[ToolResult]) -> bool:
        """Determine if the agent should stop the loop"""
        # Stop if max iterations reached
        if self.current_iteration >= self.max_iterations:
            return True
        
        # Stop if no more tool calls are needed
        message = response.get("choices", [{}])[0].get("message", {})
        if not message.get("tool_calls"):
            return True
        
        # Stop if mark_task_complete was called
        for result in tool_results:
            if any(call.get("function", {}).get("name") == "mark_task_complete" 
                   for call in message.get("tool_calls", [])):
                return True
        
        return False
    
    def run(self, user_input: str) -> Dict:
        """Run the agent with agentic loops"""
        print(f"🚀 Agent initialized successfully!")
        print(f"📝 Using model: {self.model_id}")
        print("-" * 60)
        
        # Initialize conversation with system message
        self.conversation_history = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can use tools to complete tasks. You can call multiple tools in parallel when needed."
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
        
        self.current_iteration = 0
        
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            print(f"\n🔄 Agent iteration {self.current_iteration}/{self.max_iterations}")
            
            # Get response from OpenAI (or simulation)

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=self.conversation_history,
                tools=self.tools.get_tool_definitions(),
                # tool_choice="auto",
            )
            response = response.model_dump()

            message = response["choices"][0]["message"]
            
            # Add assistant message to conversation
            self.conversation_history.append(message)
            print(f'finish reason: {response["choices"][0]["finish_reason"]}')
            print(f"💬 Agent thought: {message['content']}")
            # Check if there are tool calls to execute
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                print(f"🔧 Agent making {len(tool_calls)} tool call(s)")
                
                # Execute tool calls in parallel
                tool_results = self._execute_tool_calls_parallel(tool_calls)
                
                # Add tool results to conversation
                for result in tool_results:
                    tool_name = next(
                        (call["function"]["name"] for call in tool_calls 
                         if call["id"] == result.tool_call_id), 
                        "unknown"
                    )
                    tool_args = next(
                        (call["function"]["arguments"] for call in tool_calls 
                         if call["id"] == result.tool_call_id), 
                        "unknown"
                    )
                    print(f"🔧 Calling tool: {tool_name} with parameters: {tool_args}")
                    
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": result.content
                    })
                
                # Check if we should stop
                if self._should_stop(response, tool_results):
                    break
            else:
                # No tool calls, agent is done
                # if message.get("content"):
                #     print(f"💬 Agent: {message['content']}")
                break
        
        return {
            "success": True,
            "iterations": self.current_iteration,
            "conversation_history": self.conversation_history,
            "final_message": self.conversation_history[-1].get("content", "Task completed")
        }


def main(model_id: str):
    """Main function to run the agent"""
    print("🤖 OpenAI Agent with Tool Calling and Agentic Loops")
    print("=" * 60)
    
    
    agent = OpenAIAgent(model_id=model_id)
    
    # Example task from the image
    user_query = "search 五道口纳什, then give me the result of 4454+322-32/3 and write a txt file with what you find about him"
    
    print(f"\n👤 User: {user_query}")
    print("🧠 Agent: Thinking...")
    
    # Run the agent
    result = agent.run(user_query)
    
    print(f"\n✅ Agent completed task in {result['iterations']} iterations")
    print(f"📋 Final result: {result['final_message']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4.1-nano")
    args = parser.parse_args()
    assert load_dotenv()
    main(args.model)
