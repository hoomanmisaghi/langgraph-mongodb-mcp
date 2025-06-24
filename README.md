# langgraph-mongodb-mcp
This repo is the implemntation of a minimal langgraph chat bot that has access to mongodb-mcp enabling the user to chat with their mongo.
This implememntation expecte you to have the mongodb-mcp in docker.
The MongoDB MCP: https://github.com/mongodb-js/mongodb-mcp-server

I only did this project to challenge my skills regarding LLMs. I could not find any examples explaining how to connect this mcp to langgraph, I thaught it could be helpful. It was especially tricky to handle some weird cases of connecting and using this mcp as it kept disconnecting.

To run this you need a Gemini API key. You can easily change the llm definition to your favorite ones thanks to langchain
