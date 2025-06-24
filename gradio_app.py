import gradio as gr
import asyncio
from threading import Thread
import uuid
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
import time

# Import your MongoDB agent
from mongodb_agent import MongoDBAgent  # Adjust import path as needed


class ChatbotManager:
    """Manages multiple MongoDB agent instances for different users"""
    
    def __init__(self):
        self.agents: Dict[str, MongoDBAgent] = {}
        self.agent_tasks: Dict[str, Any] = {}
    
    async def get_or_create_agent(self, session_id: str) -> MongoDBAgent:
        """Get existing agent or create new one for session"""
        if session_id not in self.agents:
            # Create new agent for this session
            agent = MongoDBAgent(
                connection_string="mongodb://localhost:27017/IVF_human",
                max_attempts=5,
                database_name="your_database"
            )
            await agent.setup()
            await agent.build_graph()
            self.agents[session_id] = agent
        
        return self.agents[session_id]
    
    async def chat_with_agent(self, message: str, session_id: str) -> str:
        """Send message to agent and get response"""
        try:
            agent = await self.get_or_create_agent(session_id)
            human_message = HumanMessage(content=message)
            
            # Run the agent
            result = await agent.run(human_message, [])
            
            # Extract the response from the result
            if result and "messages" in result:
                messages = result["messages"]
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, 'content'):
                        return last_message.content
            
            return "I couldn't process your request. Please try again."
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def cleanup_agent(self, session_id: str):
        """Clean up agent resources"""
        if session_id in self.agents:
            try:
                self.agents[session_id].cleanup()
            except:
                pass
            del self.agents[session_id]


# Global chatbot manager
chatbot_manager = ChatbotManager()


def run_async_chat(message: str, history: list, session_id: str) -> tuple:
    """Wrapper to run async chat function in sync context"""
    
    async def async_chat():
        return await chatbot_manager.chat_with_agent(message, session_id)
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        response = loop.run_until_complete(async_chat())
    finally:
        loop.close()
    
    # Update history
    history.append([message, response])
    
    return history, ""


def create_session_id():
    """Create unique session ID for each user"""
    return str(uuid.uuid4())


def clear_chat(session_id: str):
    """Clear chat history and optionally cleanup agent"""
    # Optionally cleanup the agent (uncomment if you want fresh agent on clear)
    # chatbot_manager.cleanup_agent(session_id)
    return [], ""


# Create Gradio interface
def create_gradio_app():
    with gr.Blocks(title="MongoDB Chatbot", theme=gr.themes.Soft()) as app:
        gr.Markdown("# MongoDB Database Chatbot")
        gr.Markdown("Ask questions about your MongoDB database. Each session maintains its own connection and context.")
        
        # Session state to track user sessions
        session_id = gr.State(value=create_session_id)
        
        with gr.Row():
            with gr.Column(scale=4):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="MongoDB Assistant",
                    height=500,
                    placeholder="Start chatting with your MongoDB assistant..."
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your message",
                        placeholder="Type your MongoDB query here...",
                        container=False,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
                    new_session_btn = gr.Button("New Session", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### Tips:")
                gr.Markdown("""
                - Ask to list databases
                - Query collections
                - Get document counts
                - Find specific documents
                - Aggregate data
                
                ### Examples:
                - "List all databases"
                - "Show collections in mydb"
                - "Find documents in users collection"
                - "Count documents where age > 25"
                """)
                
                gr.Markdown("### Session Info:")
                session_display = gr.Textbox(
                    label="Session ID",
                    value=lambda: create_session_id()[:8] + "...",
                    interactive=False
                )
        
        # Event handlers
        def send_message(message, history, sess_id):
            if not message.strip():
                return history, message
            return run_async_chat(message, history, sess_id)
        
        def clear_and_reset(sess_id):
            return clear_chat(sess_id)
        
        def new_session():
            new_id = create_session_id()
            return [], "", new_id, new_id[:8] + "..."
        
        # Wire up events
        send_btn.click(
            send_message,
            inputs=[msg_input, chatbot, session_id],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            send_message,
            inputs=[msg_input, chatbot, session_id],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(
            clear_and_reset,
            inputs=[session_id],
            outputs=[chatbot, msg_input]
        )
        
        new_session_btn.click(
            new_session,
            outputs=[chatbot, msg_input, session_id, session_display]
        )
        
        # Initialize session display
        app.load(
            lambda sess_id: sess_id[:8] + "...",
            inputs=[session_id],
            outputs=[session_display]
        )
    
    return app


if __name__ == "__main__":
    # Create and launch the app
    app = create_gradio_app()
    
    # Launch with sharing enabled for multiple users
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True if you want a public link
        inbrowser=True,         # Open browser automatically
        show_error=True,        # Show detailed errors
        max_threads=40          # Support multiple concurrent users
    )