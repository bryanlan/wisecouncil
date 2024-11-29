# config.py

import os
import keys  # Assuming this is a module with API keys

# Environment variables
os.environ["OPENAI_API_KEY"] = keys.OPENAI_KEY
os.environ["ANTHROPIC_API_KEY"] = keys.CLAUDE_KEY
os.environ["TAVILY_API_KEY"] = keys.TAVILY_KEY
os.environ["GOOGLE_API_KEY"] = keys.GOOGLE_API_KEY
os.environ["XAI_API_KEY"] = keys.GROK_API_KEY
os.environ["GOOGLE_SEARCH_ID"] = keys.GOOGLE_SEARCH_ENGINE_ID
GEMINI_KEY = keys.GEMINI_API_KEY

# Constants
TOOLPREFIX = "Tool Provided Data:"
debugme = 1  # Set to 0 to disable debugging window
moderatorPromptEnd = (f"Respond in the following JSON format:\n"
    "{\n"
    '   "next_agent": "the name of the next agent to speak",\n'
    '   "agent_instruction": "Name of the next agent followed by instructions to the agent",\n'
    '   "final_thoughts": "Your final thoughts on the conversation.",\n'
    "}"
)
