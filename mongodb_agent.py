from dotenv import load_dotenv
import os
import langchain_google_genai
from typing import Annotated, List, Any, Optional, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import os
from typing import TypedDict
import sys
import asyncio

from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import Tool, Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import os
import uuid

import asyncio


load_dotenv("/mnt/hooman/llm_projects/.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    worker_attempts: int
    # React-specific fields
    current_plan: str
    task_completed: bool


class MongoDBAgent:
    """Enhanced MongoDB agent with React pattern - reasons before acting and reflects on results."""

    def __init__(self, connection_string: str = "mongodb://localhost:27017", max_attempts: int = 3, database_name: Optional[str] = None):
        self.connection_string = connection_string
        self.database_name = database_name
        self.max_attempts = max_attempts
        self.worker_llm = None
        self.tools = None
        self.graph = None
        self.agent_id = str(uuid.uuid4())
        self.memory = MemorySaver()

    async def setup(self):
        # add mcp tool
        self.tools = await self.get_mcps()
        self.worker_llm = langchain_google_genai.ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY
        )
        self.worker_llm_with_tools = self.worker_llm.bind_tools(self.tools)

    async def get_mcps(self):
        client = MultiServerMCPClient(
            {
                "mongodb": {
                    "command": "docker",
                    "args": [
                        "run",
                        "--rm",
                        "-i",
                        "--init",
                        "--network=host",
                        "mongodb/mongodb-mcp-server:latest",
                        "--readOnly",
                        "--telemetry", "disabled",
                        f"--connectionString={self.connection_string}"
                    ],
                    "transport": "stdio",
                },
            },
        )
        self.mcp_client = client
        tools = await client.get_tools()
        return tools

    def reasoner(self, state: State):
        """Plans the approach before taking action"""
        messages = state["messages"]
        
        # Get the latest human message content safely
        user_request = "general MongoDB assistance"
        for msg in reversed(messages):
            if hasattr(msg, 'content') and isinstance(msg, HumanMessage) and msg.content.strip():
                user_request = msg.content.strip()
                break
        
        # Create a proper conversation with system and human message
        reasoning_messages = [
            SystemMessage(content="""You are a MongoDB assistant. Plan your approach to help with database requests.

Think step by step:
1. What MongoDB operations are needed?
2. What information do you need first?
3. What's your approach?

Provide a brief plan (2-3 sentences) focusing on the key steps."""),
            HumanMessage(content=f"Please help me with this MongoDB request: {user_request}")
        ]
        
        response = self.worker_llm.invoke(reasoning_messages)
        
        return {
            "current_plan": response.content,
            "messages": state["messages"]
        }

    def worker(self, state: State):
        """Enhanced worker that uses the current plan"""
        current_attempts = state.get("worker_attempts", 0)
        
        if current_attempts >= self.max_attempts:
            error_message = AIMessage(
                content=f"Maximum attempts ({self.max_attempts}) reached. Please try again."
            )
            return {
                "messages": [error_message],
                "worker_attempts": current_attempts,
                "task_completed": True
            }
        
        messages = state["messages"]
        current_plan = state.get("current_plan", "")
        
        # Add plan context to system message
        if current_plan:
            plan_context = SystemMessage(content=f"Current plan: {current_plan}")
            enhanced_messages = [plan_context] + messages
        else:
            enhanced_messages = messages
        
        response = self.worker_llm_with_tools.invoke(enhanced_messages)
        
        return {
            "messages": [response],
            "worker_attempts": current_attempts + 1
        }

    def reflector(self, state: State):
        """Reflects on the results and determines if task is complete"""
        messages = state["messages"]
        
        # Check if last message has tool calls (still working) or is a final response
        if messages:
            last_message = messages[-1]
            has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls
            
            if not has_tool_calls:
                # No tool calls means we have a final response
                last_content = getattr(last_message, 'content', '') if last_message else ''
                
                reflection_messages = [
                    SystemMessage(content="""You are evaluating if a MongoDB task has been completed successfully.
                    
Analyze the response and determine:
- If it provides the requested information or completes the task: respond "COMPLETE"
- If more work is needed: respond "CONTINUE" and briefly explain what's missing"""),
                    HumanMessage(content=f"Please evaluate this response: {last_content}")
                ]
                
                reflection = self.worker_llm.invoke(reflection_messages)
                task_completed = "COMPLETE" in reflection.content.upper()
                
                return {
                    "task_completed": task_completed,
                    "messages": state["messages"]
                }
        
        # If we have tool calls, continue processing
        return {
            "task_completed": False,
            "messages": state["messages"]
        }

    def route_from_worker(self, state: State):
        """Route from worker based on tool calls and attempts"""
        current_attempts = state.get("worker_attempts", 0)
        
        if current_attempts >= self.max_attempts:
            return END
        
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
        
        return "reflector"

    def route_from_reflector(self, state: State):
        """Route from reflector based on task completion"""
        task_completed = state.get("task_completed", False)
        current_attempts = state.get("worker_attempts", 0)
        
        if task_completed or current_attempts >= self.max_attempts:
            return END
        
        return "reasoner"  # Go back to planning for next iteration

    async def build_graph(self):
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("reasoner", self.reasoner)
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("reflector", self.reflector)

        # React flow: START -> reasoner -> worker -> (tools OR reflector) -> (reasoner OR END)
        graph_builder.add_edge(START, "reasoner")
        graph_builder.add_edge("reasoner", "worker")
        
        graph_builder.add_conditional_edges(
            "worker", 
            self.route_from_worker,
            {"tools": "tools", "reflector": "reflector", END: END}
        )
        
        graph_builder.add_edge("tools", "reflector")
        
        graph_builder.add_conditional_edges(
            "reflector",
            self.route_from_reflector,
            {"reasoner": "reasoner", END: END}
        )

        self.graph = graph_builder.compile(checkpointer=self.memory)

    def cleanup(self):
        # cleanup resources if needed for tools
        pass

    async def run(self, message, history):
        """runs the whole graph"""
        config = {
            "configurable": {"thread_id": self.agent_id},
        }

        state = {
            "messages": [message],
            "worker_attempts": 0,
            "current_plan": "",
            "task_completed": False
        }

        result = await self.graph.ainvoke(state, config=config)
        print("Graph result:", result)
        return result

    async def test_tools(self):
        print("tools:", self.tools)
        
        # Test connecting to MongoDB and listing databases
        connect_tool = next((tool for tool in self.tools if tool.name == "connect"), None)
        if connect_tool:
            print("Testing connect tool...")
            connect_args = {"connectionString": self.connection_string}
            response = await connect_tool.ainvoke(connect_args)
            print("Connect tool response:", response)
        
        # Test listing databases
        list_databases_tool = next((tool for tool in self.tools if tool.name == "list-databases"), None)
        if list_databases_tool:
            print("Testing list databases tool...")
            response = await list_databases_tool.ainvoke({})
            print("List databases response:", response)
        else:
            print("List databases tool not found.")


# Example usage:
if __name__ == "__main__":
    agent = MongoDBAgent(
        connection_string="mongodb://localhost:27017", # assuming it is local MongoDB 
        max_attempts=5,
        database_name="your_database"
    )
    asyncio.run(agent.setup())
    asyncio.run(agent.build_graph())
    
    message = HumanMessage(content="list all databases and show me the collections in each")
    asyncio.run(agent.run(message, []))