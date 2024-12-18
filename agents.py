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
        
        response = openai_llm_mini.invoke(messages)
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
            f"Respond with just the name or 'END' if the conversation goals have been met or the conversation is repetitive. "
            f"Reminder: ONLY return the name of the next agent or 'END', nothing else."
        )
        
        messages = [
            SystemMessage(content=next_agent_prompt),
            *conversation_messages
        ]
        
        response = openai_llm_mini.invoke(messages)
        next_agent = response.content.strip()
        return next_agent if next_agent in self.agent_names or next_agent == 'END' else 'END'

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
        response = self.llm.invoke(messages)
        
        # Step 4: Add response to conversation
        response_message = f"{self.name}: {response.content}"
        state['conversation_messages'].append(AIMessage(content=response_message))
        
        # Step 5: Determine next agent
        next_agent = None
        if self.topology == 'last_decides_next':
            next_agent = self._determine_next_agent(state['conversation_messages'])
            state['nextAgent'] = next_agent
        elif 'moderator' in self.topology.lower() or 'moderated' in self.topology.lower():
            next_agent = "Moderator"
            state['nextAgent'] = next_agent
        
        # Add debugging info
        if 'debugging_info' not in state:
            state['debugging_info'] = []
        state['debugging_info'].append({
            'agent_name': self.name,
            'messages_sent': messages,
            'response': f"Final response: {response.content}",
            'next_agent': next_agent
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
            f"Determine how many agents are needed, what specific and complimentary roles they should play, and assign LLMs accordingly. Do not use a moderator unless one is specifically requested."
            f"Pick a topology type from the following options:\n"
            f"a) round_robin - no moderator\n"
            f"b) last_decides_next - no moderator, the last agent decides the next agent based on context\n"
            f"c) moderator_discretionary - with moderator, the moderator decides the next agent based on context\n"
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
            f"3. If the topology type includes a moderator, include 'moderator_prompt': A proposed system prompt for the moderator, which includes how the moderator should focus the discussion, the names and roles (not model type) of the agents. Be sure to use the word agent in describing the agents and be very precise in repeating any relevant setup information.\n\n"
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
