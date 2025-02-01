# agents.py

from typing import List, Union, Dict, Any, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import datetime
import json
from utils import clean_response
from llms import available_llms, cheap_llm, create_llm
from tools import Tool
from config import TOOLPREFIX, debugme, PROMPT_LENGTH_THRESHOLD
from state import AgentState

# Base Agent class to handle common functionality
class BaseAgent:
    def __init__(self, name: str, llm, role: str, available_tools: List[Tool]):
        self.name = name
        self.llm = llm
        self.role = role
        self.available_tools = available_tools

    def _invoke_tools(self, tool_calls: List[Dict], state: AgentState) -> List[Dict]:
        tool_invocation_count = state.get('tool_invocation_count', 0)
        max_tool_invocations = state.get('max_tool_invocations', float('inf'))

        # Check if we've hit the limit
        if tool_invocation_count >= max_tool_invocations:
            return []

        tool_results = []
        remaining_invocations = min(
            max_tool_invocations - tool_invocation_count,
            len(tool_calls)  # Explicitly limit to available tools
        )

        # Process only up to the remaining allowed invocations
        for call in tool_calls[:remaining_invocations]:
            tool_name = call.get('tool_name')
            tool_input = call.get('tool_input')
            tool = next((t for t in self.available_tools if t.name == tool_name), None)
            if tool:
                result = tool.invoke(tool_input)
                tool_results.append({
                    'tool_name': tool_name,
                    'tool_input': tool_input,
                    'tool_result': result
                })
                state['tool_invocation_count'] = tool_invocation_count + 1

        return tool_results

    def _process_response(self, response_text: str, state: AgentState, retry_count: int = 0) -> Tuple[Dict, List[Dict]]:
        try:
            response_data = json.loads(clean_response(response_text))

            tool_calls = response_data.get('tool_calls', [])
            tool_results = []

            if tool_calls:
                # Execute tool calls and get results
                tool_results = self._invoke_tools(tool_calls, state)

                # Create updated context as a single ToolMessage
                tool_outputs = "\\n".join([tr['tool_result'] for tr in tool_results])
                updated_context = (
                    f"Previous AI thought: {response_data.get('thought', '')}\n"
                    f"AI Tool results of thought: {tool_outputs}"
                )

                # Re-run LLM with updated context
                messages = [
                    SystemMessage(content=self.system_message),
                    *state['conversation_messages'],
                    HumanMessage(content=updated_context)
                ]

                new_response = self.llm.invoke(messages)
                final_answer = new_response.content
                response_data = json.loads(clean_response(new_response.content))

            else:
                messages = [
                    SystemMessage(content=self.system_message),
                    *state['conversation_messages'],
                ]
                final_answer = response_text

            # Add to debugging info with complete chain
            if 'debugging_info' not in state:
                state['debugging_info'] = []
            state['debugging_info'].append({
                'agent_name': self.name,
                'messages_sent': messages,
                'response': f"Final response: {final_answer}"
            })

            return response_data, tool_results

        except json.JSONDecodeError:
            return {"error": f"Error: Invalid JSON response from {self.name}"}, []
        except ValueError as e:
            return {"error": f"Error: {str(e)}"}, []

class FeedbackAgent(BaseAgent):
    def __init__(self, name: str, llm, role: str, topology: str, agent_info: List[Dict[str, str]], available_tools: List[Tool]):
        super().__init__(name, llm, role, available_tools)
        self.topology = topology
        self.agent_info = agent_info  # Full agent information including roles
        self.agent_names = [agent['name'] for agent in agent_info]  # Just the names
        self.system_message = self._construct_simple_system_message()
        
    def _construct_simple_system_message(self) -> str:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        base_message = f"You are {self.name}, {self.role}\n\n"
        base_message += f"Today's date is {current_date}.\n\n"
        
        # Add information about other agents
        base_message += "Other agents in this discussion:\n"
        for agent in self.agent_info:
            if agent['name'] != self.name:  # Skip the current agent
                base_message += f"- {agent['name']}: {agent['role']}\n"
        
        return base_message

    def _should_use_tools(self, conversation_messages) -> List[Dict]:
        """Use mini LLM to determine if and which tools to use"""
        # First check if we've hit the tool invocation limit
        if not self.available_tools:
            return []

        # Include tool information in the mini LLM prompt
        tools_info = "Available tools:\n" + "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in self.available_tools]
        )
        
        tool_decision_prompt = (
            f"{tools_info}\n\n"
            "Your task is to analyze the conversation and ONLY determine if any tools need to be used to answer the question. "
            "DO NOT continue the conversation or provide additional commentary."
        )
        
        reminder = (
            "\n\nYou must respond with ONLY a JSON object that specifies:\n"
            "1. Whether any tools need to be used to answer the question ('use_tools': true/false)\n"
            "2. If tools should be used, specify which ones in 'tool_calls'\n\n"
            "Required JSON format:\n"
            '{"use_tools": boolean, "tool_calls": [{"tool_name": "string", "tool_input": "string"}]}\n\n'
            "Example responses:\n"
            '{"use_tools": false, "tool_calls": []}\n'
            '{"use_tools": true, "tool_calls": [{"tool_name": "calculator", "tool_input": "2+2"}]}'
        )
        
        # Create a copy of the conversation messages to avoid modifying the original
        messages = conversation_messages.copy()
        
        # Add reminder to the last message if there are any messages
        if messages:
            if isinstance(messages[-1], HumanMessage):
                messages[-1] = HumanMessage(content=messages[-1].content + reminder)
            else:
                messages.append(HumanMessage(content=reminder))
        
        messages = [
            SystemMessage(content=tool_decision_prompt),
            *messages
        ]
        
        response = cheap_llm.invoke(messages)
        try:
            decision = json.loads(clean_response(response.content))
            return decision.get('tool_calls', []) if decision.get('use_tools', False) else []
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON response from tool decision: {response.content}")
            return []

    def _determine_next_agent(self, conversation_messages) -> str:
        """Use mini LLM to determine the next agent"""
        # Create a description of each agent with their role
        agent_descriptions = "\n".join([
            f"- {agent['name']}: {agent['role']}" 
            for agent in self.agent_info
        ])
        
        next_agent_prompt = (
            f"Based on the conversation and the following agents' roles:\n\n"
            f"{agent_descriptions}\n\n"
            f"Who should speak next? Consider each agent's role and expertise. "
            f"Options are: {', '.join(self.agent_names)} or 'END'. "
            f"Respond with just the exact name of the next agent or 'END' if the conversation goals have been met or the conversation is repetitive. "
            f"Reminder: ONLY return the name of the next agent or 'END', nothing else."
        )
        
        messages = [
            SystemMessage(content=next_agent_prompt),
            *conversation_messages
        ]
        
        response = cheap_llm.invoke(messages)
        next_agent = response.content.strip()
        return next_agent if next_agent in self.agent_names or next_agent == 'END' else 'END'

    # In agents.py, modify the FeedbackAgent.generate_response method:

    def generate_response(self, state: AgentState) -> AgentState:
        # Step 1: Determine if tools should be used
        tool_calls = self._should_use_tools(state['conversation_messages'])
        
        # Step 2: Execute tools if needed and collect results
        tool_results = []
        if tool_calls:
            tool_results = self._invoke_tools(tool_calls, state)
            # Add tool results to conversation context
            for result in tool_results:
                tool_message = f"{TOOLPREFIX}{result['tool_result']}"
                state['conversation_messages'].append(AIMessage(content=tool_message))
        
        # Step 3: Generate main response using primary LLM
        messages = [
            SystemMessage(content=self.system_message),
            *state['conversation_messages']
        ]
        
        # Add a reminder to ensure response
        reminder_message = (
            "\n\nIMPORTANT: You must provide a response that continues the conversation. "
            "If tool results are present, incorporate them into your response. "
            "If no tool results are present or they're not helpful, provide your best response based on your knowledge. "
            "Never return an empty response."
        )
        
        messages.append(HumanMessage(content=reminder_message))
        
        response = self.llm.invoke(messages)
        
        # Step 4: Add response to conversation
        # Handle case where response.content might be empty or a list
        final_response_text = response.content
        if isinstance(final_response_text, list):
            final_response_text = ' '.join(str(item) for item in final_response_text)
        
        # If response is empty or whitespace, generate a fallback response
        if not final_response_text.strip():
            fallback_messages = [
                SystemMessage(content=self.system_message),
                *state['conversation_messages'],
                HumanMessage(content=(
                    "The previous response was empty. As an expert in your role, "
                    "please provide a meaningful response to continue the conversation. "
                    "If tool results were provided, incorporate them. If not, use your expertise."
                ))
            ]
            fallback_response = self.llm.invoke(fallback_messages)
            final_response_text = fallback_response.content
        
        # Ensure we have some content
        if not final_response_text.strip():
            final_response_text = (
                f"{self.name}: Based on the conversation context and my role, "
                "I'll continue the discussion. "
            )
        
        # Prevent duplicate agent name prefix
        if not final_response_text.strip().lower().startswith(self.name.lower()):
            final_response_text = f"{self.name}: {final_response_text}"
        
        state['conversation_messages'].append(AIMessage(content=final_response_text))
        
        # Handle next agent selection based on topology
        if self.topology == 'last_decides_next':
            next_agent = self._determine_next_agent(state['conversation_messages'])
            # If there's a moderator and agent tries to end, return to moderator instead
            if next_agent == 'END' and state.get('moderatorName'):
                state['nextAgent'] = state['moderatorName']
            else:
                state['nextAgent'] = next_agent
        elif self.topology == 'round_robin':
            current_idx = self.agent_names.index(self.name)
            next_idx = (current_idx + 1) % len(self.agent_names)
            state['nextAgent'] = self.agent_names[next_idx]
        elif 'moderator' in self.topology.lower() or 'moderated' in self.topology.lower():
            state['nextAgent'] = state['moderatorName']
        
        # Add debugging info
        if 'debugging_info' not in state:
            state['debugging_info'] = []
        state['debugging_info'].append({
            'agent_name': self.name,
            'messages_sent': messages,
            'response': f"Final response: {response.content}",
            'next_agent': state['nextAgent']
        })
        
        return state

class ModeratorAgent(BaseAgent):
    def __init__(self, llm, available_tools: List[Tool]):
        super().__init__("Moderator", llm, "", available_tools)
        self.setup = None

    def set_setup(self, setup: Dict[str, Any]):
        self.setup = setup
        self.system_message = setup['moderator_prompt']
        self.agent_names = [agent['name'] for agent in setup['agents']]

    def _process_response(self, response_text: str, state: AgentState, retry_count: int = 0) -> Tuple[Dict, List[Dict]]:
        MAX_RETRIES = 3
        response_data, tool_results = super()._process_response(response_text, state, retry_count)
        if 'error' in response_data:
            return response_data, tool_results
        # For Moderator, expected keys are 'next_agent', 'agent_instruction', 'final_thoughts'
        required_keys = ['next_agent', 'agent_instruction', 'final_thoughts']
        missing_keys = [key for key in required_keys if key not in response_data]
        if missing_keys:
            if retry_count < MAX_RETRIES:
                return self._process_response(response_text, state, retry_count + 1)
            else:
                return {"error": f"Error: Missing keys {missing_keys} after {MAX_RETRIES} attempts"}, tool_results
        return response_data, tool_results

    def generate_response(self, state: AgentState) -> AgentState:
        messages = [SystemMessage(content=self.system_message)] + state['conversation_messages']

        # If prompt length too large, remind to ONLY respond in JSON
        total_length = sum(len(m.content) for m in messages)
        if total_length > PROMPT_LENGTH_THRESHOLD:
            messages.append(HumanMessage(
                content="IMPORTANT REMINDER: You must ONLY respond in JSON. "
                        "If your JSON is invalid or you provide additional commentary, the request fails."
            ))

        response = self.llm.invoke(messages)

        max_retries = 3
        attempt = 0

        response_data = None
        tool_results = None
        while attempt < max_retries:
            try:
                # Process response and handle any tool calls
                response_data, tool_results = self._process_response(response.content, state)
                break  # Exit the loop if successful
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed in _process_response: {e}")
                
                if attempt < max_retries:
                    print("Retrying...")
                    response = self.llm.invoke(messages)  # Reinvoke the LLM
                else:
                    print("Max retries reached. Raising the exception.")
                    raise  # Re-raise the exception if max retries are exceeded

        if 'error' in response_data:
            # Handle error
            state['conversation_messages'].append(
                AIMessage(content=f"{self.name}: {response_data['error']}")
            )
            state['nextAgent'] = 'END'
            return state

        # Add tool results to conversation
        if tool_results:
            for tool_result in tool_results:
                tool_message = f"{TOOLPREFIX}{tool_result['tool_result']}"
                state['conversation_messages'].append(AIMessage(content=tool_message))

        # Validate next_agent
        next_agent = response_data.get('next_agent', 'END')
        if next_agent not in self.agent_names and next_agent != 'END':
            next_agent = 'END'

        if next_agent == 'END' and response_data.get('final_thoughts'):
            state['conversation_messages'].append(
                AIMessage(content=f"{self.name}: {response_data['final_thoughts']}")
            )
            state['nextAgent'] = 'END'
        else:
            state['conversation_messages'].append(
                HumanMessage(content=f"{self.name}: {response_data['agent_instruction']}")
            )
            state['nextAgent'] = next_agent

        return state

# Setup Agent Class
class SetupAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate_setup(self, setup_info: str) -> Dict[str, Any]:
        # Prepare the list of available LLMs as a string
        llm_descriptions = "\n".join([f"{llm['name']}: {llm['description']}" for llm in available_llms])

        # Prepare the prompt
        prompt = (
            f"You are a setup agent. Based on the following setup information: '{setup_info}', and the available agents:\n{llm_descriptions}\n\n"
            f"Determine how many agents are needed, what specific and complimentary roles they should play, and assign LLMs accordingly. Do not use a moderator unless one is specifically requested.\n\n"
            f"Provide a JSON configuration that includes:\n"
            f"1. 'agents': A list of agents where:\n"
            f"   - If a moderator is requested, the first agent MUST be the moderator agent with:\n"
            f"     * 'name': 'Moderator'\n"
            f"     * 'type': The LLM to use for the moderator\n"
            f"     * 'role': A system prompt describing the moderator's role\n"
            f"   - For all other agents:\n"
            f"     * 'name': The agent's friendly name\n"
            f"     * 'type': The LLM to use (from the available agents)\n"
            f"     * 'role': The system prompt for the agent that describes what you think the agent should do to contribute to the conversation\n"
            f"2. If a moderator is requested, include 'moderator_prompt': A proposed system prompt for the moderator, which MUST include 1. How the moderator should focus the discussion, be very precise in repeating any relevant setup information. 2. Agent names (NOT model type)followed by their roles (not model type). \n\n"
            f"Please return the JSON without any additional text or formatting."
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            # Clean response content by removing code fences
            cleaned_content = clean_response(response.content)
            setup_data = json.loads(cleaned_content)
            

            return setup_data
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to parse setup data: {e}")

