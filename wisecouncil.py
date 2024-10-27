from typing import List, Union, TypedDict, Dict, Any, Tuple
from langgraph.graph import Graph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import gradio as gr
import os
import keys
import json
import re
import copy

# Debugging flag
debugme = 1  # Set to 0 to disable debugging window
moderatorPromptEnd = " Please instruct the best next agent to speak by providing a JSON response with 'next_agent' which names the next agent, using END if the conversation goals are met,'agent_instruction' which addresses the agent by name and guides the conversation, and 'final_thoughts' which sums up the conversation.  ONLY respond with JSON"

def clean_response(content: str) -> str:
    """
    Cleans the response content by removing code fences and any leading/trailing whitespace.
    """
    # Remove any backticks and "json" label from the response
    clean_content = content.strip().strip('```').replace("json\n", "").replace("```", "")
    return clean_content

# Ensure you have set these environment variables
os.environ["OPENAI_API_KEY"] = keys.OPENAI_KEY
os.environ["ANTHROPIC_API_KEY"] = keys.CLAUDE_KEY

# Define the state
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

# Initialize LLMs
openai_llm = ChatOpenAI(model="gpt-4o", temperature=0)
claude_llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0)

# Available LLMs with descriptions
available_llms = [
    {
        "name": "openai_gpt4",
        "description": "OpenAI's GPT-4 model, a best-in-class foundation model capable of general use.",
        "llm": openai_llm
    },
    {
        "name": "anthropic_claude",
        "description": "Anthropic's Claude model, a best-in-class foundation model capable of general use.",
        "llm": claude_llm
    }
]

# Feedback Agent Class
class FeedbackAgent:
    def __init__(self, name: str, llm, role: str, topology: str, agent_names: List[str]):
        self.name = name
        self.llm = llm
        self.role = role
        self.topology = topology
        self.agent_names = agent_names
        self.system_message = self.construct_system_message()

    def construct_system_message(self) -> str:
        base_message = self.role + "\n\n"
        if self.topology == 'last_decides_next':
            base_message += (
                "Please also return next_agent: [name of next agent to respond] based on who you think "
                "will be best able to add to the discussion at this time, given the following agents and their capabilities:\n"
            )
            for agent in self.agent_names:
                base_message += f"- {agent['name']}: {agent['role']}\n"
        return base_message

    def generate_response(self, state: AgentState) -> AgentState:
        # Begin with the system message
        messages = [SystemMessage(content=self.system_message)] + state['conversation_messages']
       
        response = self.llm.invoke(messages)
        response_text = response.content.strip()
        if not response_text.startswith(self.name + ":"):
            response_text = self.name + ":" + response_text

        newMessage = AIMessage(content=response_text)


        # Add response to conversation
        state['conversation_messages'].append(newMessage)
        
        # Collect debugging info
        if 'debugging_info' not in state:
            state['debugging_info'] = []
        state['debugging_info'].append({
            'agent_name': self.name,
            'messages_sent': messages,
            'response': response
        })
        
  

        if self.topology == 'last_decides_next':
            # Extract next_agent from the response using regex
            match = re.search(r'next_agent\s*:\s*(\w+)', response_text, re.IGNORECASE)
            if match:
                next_agent = match.group(1)
                # Validate next_agent
                if any(agent['name'] == next_agent for agent in state['agents']):
                    state['saved_next_agent'] = next_agent
                    # Remove the 'next_agent' part from the response
                    cleaned_response = re.sub(r'next_agent\s*:\s*\w+', '', response_text, flags=re.IGNORECASE).strip()
                    # Update the last AIMessage with cleaned response
                    state['conversation_messages'][-1] = AIMessage(content=cleaned_response)
                    # Add HumanMessage to address the next agent
                    state['conversation_messages'].append(
                        HumanMessage(content=f"{next_agent} Please continue the conversation.")
                    )
                else:
                    # Invalid next_agent, proceed to next in agent_order
                    if state.get('agent_order'):
                        next_agent = state['agent_order'].pop(0)
                        state['saved_next_agent'] = next_agent
                        state['conversation_messages'].append(
                            HumanMessage(content=f"{next_agent} Please continue the conversation.")
                        )
                    else:
                        state['nextAgent'] = 'END'
            else:
                # If no next_agent found, proceed to next in agent_order
                if state.get('agent_order'):
                    next_agent = state['agent_order'].pop(0)
                    state['saved_next_agent'] = next_agent
                    state['conversation_messages'].append(
                        HumanMessage(content=f"{next_agent} Please continue the conversation.")
                    )
                else:
                    state['nextAgent'] = 'END'
        elif self.topology == 'round_robin':
            if state.get('agent_order'):
                next_agent = state['agent_order'].pop(0)
                state['conversation_messages'].append(
                    HumanMessage(content=f"{next_agent} Please continue the conversation.")
                )
            else:
                state['nextAgent'] = 'END'
        # For other topologies, handled by Moderator
        else:
            state['nextAgent'] = state['moderatorName']
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
            f"2. 'agents': A list of agents (including the Moderator agent, if applicable), each with:\n"
            f"   - 'name': The agent's friendly name.\n"
            f"   - 'type': The LLM to use (from the available agents).\n"
            f"   - 'role': The system prompt for the agent that describes who the agent is. \n"
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
            f"Generate a proposed system prompt for the moderator, which includes the goal for the discussion, the goal for feedback agent participation order, and how to instruct the agents to return a JSON response with 'next_agent' and 'agent_instruction'.\n\n"
            f"Please return the moderator prompt without any additional text or formatting."
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        moderator_prompt = clean_response(response.content)
        return moderator_prompt

# Moderator Agent Class
class ModeratorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.setup = None

    def set_setup(self, setup: Dict[str, Any]):
        self.setup = setup
        self.system_message = setup['moderator_prompt']
        self.agent_names = [agent['name'] for agent in setup['agents']]
        self.current_index = 0  # For moderated round robin

    def generate_response(self, state: AgentState) -> AgentState:
        # Use the moderator prompt as system message
        messages = [SystemMessage(content=self.system_message)] + copy.deepcopy(state['conversation_messages'])
        # Append the moderatorPromptEnd to the last HumanMessage
        if messages and isinstance(messages[-1], HumanMessage):
            messages[-1].content += moderatorPromptEnd

        # Invoke the LLM
        response = self.llm.invoke(messages)
        # Clean response content by removing code fences
        cleaned_content = clean_response(response.content)
        response_content = json.loads(cleaned_content)
        if state['nextAgent'].upper()== 'END' and response_content['final_thoughts'] is not None:
            state['conversation_messages'].append(AIMessage(state['moderatorName']+":"+response_content['final_thoughts']))
            state['nextAgent']='END'
        else:
            # Add agent_instruction to conversation_messages as AIMessage
            agent_instruction = response_content.get('agent_instruction', '')
            state['conversation_messages'].append(HumanMessage(state['moderatorName']+":"+agent_instruction))
            state['nextAgent'] = response_content.get('next_agent', 'END')
        
        # Collect debugging info
        if 'debugging_info' not in state:
            state['debugging_info'] = []
        state['debugging_info'].append({
            'agent_name': state['moderatorName'],  # Assuming moderator name is 'Moderator'
            'messages_sent': messages,
            'response': response
        })

        return state

# Initialize the SetupAgent
setup_agent = SetupAgent(openai_llm)
# Moderator agent will be initialized after setup

def should_continue(state: AgentState) -> str:
    nextState = state.get('nextAgent', 'END')
    if nextState.upper() =='END':
        nextState = END
    return nextState

def format_messages(messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) -> str:
    return "\n\n".join([f"{m.content}" for m in messages])

def setup_conversation(setup_info: str):
    # Generate setup using SetupAgent
    try:
        setup_data = setup_agent.generate_setup(setup_info)
    except ValueError as e:
        return "Error in setup generation: " + str(e)
    
    topology_type = setup_data.get('topology_type', '')
    agents = setup_data.get('agents', [])
    moderator_prompt = setup_data.get('moderator_prompt', '')
    if moderator_prompt  != '':
        moderator_prompt += moderatorPromptEnd
    
    # Convert agents to list of lists for DataFrame
    agents_df = [[agent['name'], agent['type'], agent['role']] for agent in agents]
    
    return topology_type, agents_df, moderator_prompt

def recalculate_moderator_prompt(setup_info: str, topology_type: str, agents_df: List[List[str]]) -> str:
    # Convert agents_df to list of dicts
    agents = []
    for row in agents_df:
        if len(row) >= 3:
            agents.append({'name': row[0], 'type': row[1], 'role': row[2]})
    # Recalculate the moderator prompt
    if not agents:
        return ""
    moderator_prompt = setup_agent.recalculate_moderator_prompt(setup_info, topology_type, agents)
    return moderator_prompt

def run_conversation(
    setup_info: str,
    council_question: str,
    topology_type: str,
    agents_df: List[List[str]],
    moderator_prompt: str,
    max_iterations: int
) -> Tuple[str, str]:
    # Convert agents_df to list of dicts
    agents_rows = agents_df.values.tolist()
    agents_list = [{'name': str(row[0]).strip(), 'type': str(row[1]).strip(), 'role': str(row[2]).strip()} 
                   for row in agents_rows if len(row) >= 3]
    # Ensure agents_list is valid before proceeding
    if not agents_list:
        return "Error: No valid agents provided.", ""

    # Create agent instances, differentiating between FeedbackAgents and ModeratorAgent
    agent_objects = {}
    moderator_agent = None
    moderatorName = ""

    for agent_info in agents_list:
        name = agent_info['name']
        llm_type = agent_info['type']
        role = agent_info['role']
        isModerator = ('moderator' in role.lower() or 'moderator' in name.lower() or 'moderate' in role.lower())

        # Find the LLM object
        llm = next((available_llm['llm'] for available_llm in available_llms if available_llm['name'] == llm_type), None)

        if llm is None:
            return f"Error: LLM '{llm_type}' not found for agent '{name}'", ""

        # Check if the agent is the moderator or a feedback agent based on the topology
        if topology_type in ['moderator_discretionary', 'moderated_round_robin'] and isModerator:
            # Create ModeratorAgent for the agent with the moderator role
            moderator_agent = ModeratorAgent(llm)
            moderator_agent.set_setup({
                'topology_type': topology_type,
                'agents': agents_list,
                'moderator_prompt': moderator_prompt
            })
            moderatorName = name
        else:
            # Create FeedbackAgent for regular feedback agents
            agent_objects[name] = FeedbackAgent(
                name=name,
                llm=llm,
                role=role,
                topology=topology_type,
                agent_names=agents_list
            )
    
    # Error if no moderator found when expected
    if topology_type in ['moderator_discretionary', 'moderated_round_robin'] and moderator_agent is None:
        return "Error: No moderator found in agents list.", ""

    # Build the workflow graph based on the topology type
    workflow = Graph()

    if topology_type == 'round_robin':
        # Generate agent_order based on max_iterations
        agent_order = [agent['name'] for agent in agents_list]
        agent_order *= max_iterations

        # Add FeedbackAgent nodes
        for agent_name, feedback_agent in agent_objects.items():
            workflow.add_node(agent_name, feedback_agent.generate_response)
        
        # Add edges in round-robin fashion
        agent_names = list(agent_objects.keys())
        for i, name in enumerate(agent_names):
            next_name = agent_names[(i + 1) % len(agent_names)]
            workflow.add_edge(name, next_name)

    elif topology_type == 'last_decides_next':
        # Add FeedbackAgent nodes and conditional edges
        first_agent_name = agent_objects.keys()[0]
        for agent_name, feedback_agent in agent_objects.items():
            workflow.add_node(agent_name, feedback_agent.generate_response)
            workflow.add_conditional_edges(agent_name, should_continue)
    
    elif topology_type in ['moderator_discretionary', 'moderated_round_robin']:
        # Add the moderator node and FeedbackAgent nodes
        workflow.add_node(moderatorName, moderator_agent.generate_response)
        
        # Add FeedbackAgent nodes and conditional edges
        for agent_name, feedback_agent in agent_objects.items():
            workflow.add_node(agent_name, feedback_agent.generate_response)
        
        workflow.add_conditional_edges(moderatorName, should_continue)
        for agent_name in agent_objects.keys():
            workflow.add_conditional_edges(agent_name, should_continue)
    
    else:
        return f"Error: Unknown topology type '{topology_type}'", ""

    # Set the entry point
    if topology_type in ['moderator_discretionary', 'moderated_round_robin']:
        workflow.set_entry_point(moderatorName)
    elif topology_type in ['round_robin']:
        # Start with the first agent in agent_order for round robin or last_decides_next
        if agents_list:
            first_agent_name = agents_list[0]['name']
            workflow.set_entry_point(first_agent_name)
            #remove the first agent so that the appropriate next agent will be used
            agent_order.pop(0)
        else:
            return "Error: No agents defined.", ""

    # Compile the chain and process the conversation
    chain = workflow.compile()

    # Initialize state
    state: AgentState = {
        "conversation_messages": [HumanMessage(content=council_question)],
        "setup_messages": [],
        "current_agent": "",
        "nextAgent": "",
        "agentSysPrompt": "",
        "agent_order": agent_order if topology_type == 'round_robin' else [],
        "topology": topology_type,
        "agents": agents_list,
        "control_flow_index": 0,
        "debugging_info": [],
        "moderatorName": moderatorName if moderatorName else ""
    }

    output_conversation = ""
  
    
    for i, s in enumerate(chain.stream(state)):
        if i >= max_iterations:
            break

        # Collect messages
        current_agent = list(s.keys())[0]
        agent_state = s[current_agent]
        # Use conversation_messages to build the output
        messages = agent_state.get('conversation_messages', [])
        output_conversation = format_messages(messages)
        output_debugging = ""
        # Collect debugging info
        if debugme == 1 and 'debugging_info' in agent_state:
            for debug_entry in agent_state['debugging_info']:
                agent_name = debug_entry['agent_name']
                messages_sent = debug_entry['messages_sent']
                response = debug_entry['response']
                output_debugging += f"\n\nAgent: {agent_name}\nMessages Sent:\n"
                output_debugging += format_messages(messages_sent)
                output_debugging += f"\n\nResponse:\n{response.content}"

    return output_conversation, output_debugging

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Wise Council")

    with gr.Row():
        setup_info_input = gr.Textbox(label="Setup Information", lines=2)
        council_question_input = gr.Textbox(label="Question for the Council", lines=2)
        setup_button = gr.Button("Generate Setup")

    with gr.Row():
        topology_dropdown = gr.Dropdown(
            choices=['round_robin', 'last_decides_next', 'moderator_discretionary', 'moderated_round_robin'],
            label="Topology Type"
        )
    
    with gr.Row():
        agents_dataframe = gr.Dataframe(
            headers=['name', 'type', 'role'],
            datatype=['str', 'str', 'str'],
            interactive=True,
            label="Agents (Name, Type, Role)"
        )
    
    with gr.Row():
        moderator_prompt_textbox = gr.Textbox(
            label="Moderator Prompt",
            lines=5
        )
        recalculate_button = gr.Button("Recalculate Moderator Prompt")

    with gr.Row():
        max_iterations_input = gr.Slider(
            minimum=1,
            maximum=20,
            step=1,
            value=5,
            label="Max Iterations"
        )
        run_button = gr.Button("Begin Conversation")

    conversation_output = gr.Textbox(label="Conversation Output", lines=20)
    debugging_output = gr.Textbox(label="Debugging Output", lines=20, visible=debugme)

    # Function to populate the setup fields after generating setup
    def populate_setup_fields(setup_result):
        if isinstance(setup_result, str):
            return gr.update(), gr.update(), gr.update(), setup_result
        topology_type, agents_df, moderator_prompt = setup_result
        return topology_type, agents_df, moderator_prompt

    setup_button.click(
        fn=setup_conversation,
        inputs=setup_info_input,
        outputs=[topology_dropdown, agents_dataframe, moderator_prompt_textbox]
    )

    recalculate_button.click(
        fn=recalculate_moderator_prompt,
        inputs=[setup_info_input, topology_dropdown, agents_dataframe],
        outputs=moderator_prompt_textbox
    )

    run_button.click(
        fn=run_conversation,
        inputs=[setup_info_input, council_question_input, topology_dropdown, agents_dataframe, moderator_prompt_textbox, max_iterations_input],
        outputs=[conversation_output, debugging_output]
    )

demo.launch(share=False)
