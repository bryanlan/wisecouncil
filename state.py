# state.py

from typing import TypedDict, List, Union, Dict, Any
from llms import HumanMessage, AIMessage, SystemMessage

class AgentState(TypedDict, total=False):
    # conversation_messages holds the full conversational context
    conversation_messages: List[Union[SystemMessage, HumanMessage, AIMessage]]
    setup_messages: List[Union[SystemMessage, HumanMessage, AIMessage]]
    current_agent: str
    nextAgent: str
    agentSysPrompt: str
    agent_order: List[str]
    topology: str
    agents: List[Dict[str, Any]]
    control_flow_index: int
    proposed_setup: Dict[str, Any]
    moderator_prompt: str
    saved_next_agent: str
    # debugging_info
    debugging_info: List[Dict[str, Any]]
    # Removed 'last_conversation'
    moderatorName: str
    tool_invocation_count: int  # Track number of tool uses
    max_tool_invocations: int   # Store the limit
