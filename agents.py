# agents.py

from typing import List, Union, Dict, Any, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import datetime
import json
from utils import clean_response
from llms import openai_llm, openai_llm_mini, available_llms
from tools import Tool
from config import TOOLPREFIX, debugme
from state import AgentState

# Base Agent class to handle common functionality
class BaseAgent:
    def __init__(self, name: str, llm, role: str, available_tools: List[Tool]):
        self.name = name
        self.llm = llm
        self.role = role
        self.available_tools = available_tools
        self.system_message = self._construct_system_message()

    def _construct_system_message(self) -> str:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        base_message = f"You are {self.name}, {self.role}\n\n"
        base_message += f"Today's date is {current_date}.\n"
        if self.available_tools:
            base_message += "Available tools:\n"
            for tool in self.available_tools:
                base_message += f"- {tool.name}: {tool.description}\n"
        base_message += "\nYou must respond with ONLY a valid JSON object using the following strict formatting rules:\n"
        base_message += "1. Use double quotes (not single quotes) for all JSON keys and string values\n"
        base_message += "2. For newlines in strings, use the escaped sequence '\\n' (not actual line breaks)\n"
        base_message += "3. The JSON object must contain these exact keys:\n"
        base_message += "   - 'thought': your reasoning process as a single-line string\n"
        base_message += "   - 'response': your response as a single-line string (use '\\n' for paragraph breaks)\n"
        base_message += "   - 'tool_calls': (optional) array of objects with 'tool_name' and 'tool_input' strings\n"
        base_message += "4. Do not include any markdown formatting or code blocks\n"
        base_message += "5. Do not wrap the JSON in quotes or backticks\n"
        base_message += "Example format:\n"
        base_message += '{"thought":"reasoning here","response":"first paragraph\\n\\nsecond paragraph","tool_calls":[]}\n'
        base_message += "Ensure your response is exactly one JSON object that can be parsed by JSON.parse()\n"
        return base_message

    def _invoke_tools(self, tool_calls: List[Dict], state: AgentState) -> List[Dict]:
        tool_results = []
        for call in tool_calls:
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
    def __init__(self, name: str, llm, role: str, topology: str, agent_names: List[str], available_tools: List[Tool]):
        super().__init__(name, llm, role, available_tools)
        self.topology = topology
        self.agent_names = agent_names
        if topology == 'last_decides_next':
            self.system_message += "\n- 'next_agent': the name of the next agent to speak"

    def _process_response(self, response_text: str, state: AgentState, retry_count: int = 0) -> Tuple[Dict, List[Dict]]:
        MAX_RETRIES = 3
        response_data, tool_results = super()._process_response(response_text, state, retry_count)
        if 'error' in response_data:
            return response_data, tool_results
        if 'response' not in response_data:
            if retry_count < MAX_RETRIES:
                return self._process_response(response_text, state, retry_count + 1)
            else:
                return {"error": f"Error: No 'response' key found after {MAX_RETRIES} attempts"}, tool_results
        return response_data, tool_results

    def generate_response(self, state: AgentState) -> AgentState:
        messages = [SystemMessage(content=self.system_message)] + state['conversation_messages']
        response = self.llm.invoke(messages)

        # Process response and handle any tool calls
        response_data, tool_results = self._process_response(response.content, state)

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

        # Add final response to conversation
        response_message = f"{self.name}: {response_data['response']}"
        state['conversation_messages'].append(HumanMessage(content=response_message))

        # Handle next agent selection based on topology
        if self.topology == 'last_decides_next':
            next_agent = response_data.get('next_agent')
            if next_agent and next_agent in [agent for agent in self.agent_names]:
                state['saved_next_agent'] = next_agent
            else:
                state['nextAgent'] = 'END'
        elif 'moderator' in self.topology.lower():
            state['nextAgent'] = "Moderator"

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
        response = self.llm.invoke(messages)

        # Process response and handle any tool calls
        response_data, tool_results = self._process_response(response.content, state)

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

        if state['nextAgent'].upper() == 'END' and response_data.get('final_thoughts'):
            state['conversation_messages'].append(
                AIMessage(content=f"{self.name}: {response_data['final_thoughts']}")
            )
            state['nextAgent'] = 'END'
        else:
            state['conversation_messages'].append(
                HumanMessage(content=f"{self.name}: {response_data['agent_instruction']}")
            )
            state['nextAgent'] = response_data.get('next_agent', 'END')

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
            f"Determine how many agents are needed, what specific and complimentary roles they should play, and assign LLMs accordingly. Do not use a moderator unless one is specifically requested."
            f"Pick a topology type from the following options:\n"
            f"a) round_robin - no moderator\n"
            f"b) last_decides_next - no moderator\n"
            f"c) moderator_discretionary - with moderator\n"
            f"d) moderated_round_robin - with moderator\n\n"
            f"Provide a JSON configuration that includes:\n"
            f"1. 'topology_type': One of 'round_robin', 'last_decides_next', 'moderator_discretionary', 'moderated_round_robin'.\n"
            f"2. 'agents': A list of agents where:\n"
            f"   - If topology_type is 'moderator_discretionary' or 'moderated_round_robin', the first agent MUST be the moderator agent with:\n"
            f"     * 'name': 'Moderator'\n"
            f"     * 'type': The LLM to use for the moderator\n"
            f"     * 'role': A system prompt describing the moderator's role\n"
            f"   - For all other agents (and all agents in non-moderated topologies):\n"
            f"     * 'name': The agent's friendly name\n"
            f"     * 'type': The LLM to use (from the available agents)\n"
            f"     * 'role': The system prompt for the agent that describes who the agent is\n"
            f"3. If the topology type includes a moderator, include 'moderator_prompt': A proposed system prompt for the moderator, which includes how the moderator should focus the discussion, the names and roles (not model type) of the agents, the goal for feedback agent participation order.\n\n"
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

    def recalculate_moderator_prompt(self, setup_info: str, topology_type: str, agents: List[Dict[str, Any]]) -> str:
        if topology_type not in ['moderator_discretionary', 'moderated_round_robin']:
            return ""

        # Prepare the agents list as a string
        agents_str = "\n".join([f"{agent['name']}: {agent['role']}" for agent in agents])

        # Prepare the prompt
        prompt = (
            f"You are a setup agent. Based on the following setup information: '{setup_info}', the topology type '{topology_type}', and the agents:\n{agents_str}\n\n"
            f"Generate a proposed system prompt for the moderator, which includes the goal for the discussion, the goal for feedback agent participation order.\n\n"
            f"Please return the moderator prompt without any additional text or formatting."
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        moderator_prompt = clean_response(response.content)
        return moderator_prompt
