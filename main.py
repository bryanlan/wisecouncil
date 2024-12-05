# main.py
import gradio as gr
from typing import List, Dict, Any, Tuple
from langgraph.graph import Graph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from llms import available_llms

from config import debugme, TOOLPREFIX, moderatorPromptEnd
from utils import clean_response, format_messages, should_continue
from llms import openai_llm
from agents import SetupAgent, FeedbackAgent, ModeratorAgent
from tools import ResearchTool
from state import AgentState

# Initialize the SetupAgent
setup_agent = SetupAgent(openai_llm)

# Function definitions (setup_conversation, recalculate_moderator_prompt, run_conversation)
# Include your functions here, making sure to import any necessary components from the modules.

# Initialize the SetupAgent
setup_agent = SetupAgent(openai_llm)
# Moderator agent will be initialized after setup


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
    max_iterations: int,
    suppress_webpage_popups:bool, 
    max_image_per_webpage: int, 
    max_total_image_interpretations:int,
    suppressResearch:bool,
    prepare_report: bool,
    reporterPrompt: str
) -> Tuple[str, str,str]:
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
    available_tools = [ResearchTool(dismissPopups=suppress_webpage_popups,
                                    max_interpretations_per_page=max_image_per_webpage, 
                                    max_total_image_interpretations=max_total_image_interpretations)]

    for agent_info in agents_list:
        name =   agent_info['name']
        llm_type = agent_info['type']
        role = agent_info['role']
        isModerator = ('moderator' in name.lower())

        # Find the LLM object
        llm = next((available_llm['llm'] for available_llm in available_llms if available_llm['name'] == llm_type), None)

        if llm is None:
            return f"Error: LLM '{llm_type}' not found for agent '{name}'", ""

        # Check if the agent is the moderator or a feedback agent based on the topology
        if topology_type in ['moderator_discretionary', 'moderated_round_robin'] and isModerator:
            # Create ModeratorAgent for the agent with the moderator role
            moderator_agent = ModeratorAgent(llm, available_tools)
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
                agent_names=agents_list,
                available_tools= available_tools
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
        first_agent_name = next(iter(agent_objects.keys()))
        for agent_name, feedback_agent in agent_objects.items():
            workflow.add_node(agent_name, feedback_agent.generate_response)
            workflow.add_conditional_edges(agent_name, should_continue)
        workflow.set_entry_point(first_agent_name)
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
  
    config = {"recursion_limit": 50}
    for i, s in enumerate(chain.stream(state, config=config )):
        if i >= max_iterations:
            break
        if "__end__" in s or state['nextAgent'] == 'none':
            break

        # Collect messages
        current_agent = list(s.keys())[0]
        agent_state = s[current_agent]
        # Use conversation_messages to build the output
        messages = agent_state.get('conversation_messages', [])
        output_conversation = format_messages(messages, suppressResearch)
        output_debugging = ""
        # Collect debugging info
        if debugme == 1 and 'debugging_info' in agent_state:
            for debug_entry in agent_state['debugging_info']:
                agent_name = debug_entry['agent_name']
                messages_sent = debug_entry['messages_sent']
                response = debug_entry['response']
                output_debugging += f"\n\nAgent: {agent_name}\nMessages Sent:\n"
                output_debugging += format_messages(messages_sent, False)
                output_debugging += f"\n\nResponse:\n{response}"

    if prepare_report:
        # Construct the conversation transcript
        conversation_transcript = format_messages(state['conversation_messages'], suppressToolMsg=False)
        # Replace placeholder in reporterPrompt with actual council_question
        report_prompt = reporterPrompt.replace("[council question]", council_question)
        # Prepare the prompt for the report
        report_prompt = (
            f"{report_prompt}\n\n"
            f"Conversation and Tool Outputs:\n{conversation_transcript}"
        )
        # Generate the report using the LLM
        report_response = openai_llm.invoke([HumanMessage(content=report_prompt)])
        report = report_response.content.strip()
    else:
        report = ""


    return output_conversation, output_debugging, report

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Wise Council")
    reporterPrompt = '''Using the following conversation and tool outputs, create a research report designed to answer the question: [council question].\n The report should be very detailed and scientific, presenting uncertainty in a probabilistic fashion, and if necessary, contain subjects for further research or next steps, as appropriate.'''
    with gr.Row():
        setup_info_input = gr.Textbox(label="Setup Information", lines=2)
        council_question_input = gr.Textbox(label="Question for the Council", lines=2)
        report_instructions = gr.Textbox(label="Prompt for reporter", value=reporterPrompt, lines=2)

    with gr.Row():
        suppress_webpage_popups = gr.Checkbox(label="Suppress Webpage Popups (very slow)", value=False)
        max_image_per_webpage = gr.Number(label="Max Image Interpretations per Webpage", value=4)
        max_total_image_interpretations = gr.Number(label="Max Total Image Interpretations", value=10)
        suppress_research = gr.Checkbox(label="Suppress Research in Chat", value=False)
        prepare_report = gr.Checkbox(label="Prepare Report", value=False)
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
            maximum=40,
            step=1,
            value=5,
            label="Max Iterations"
        )
        run_button = gr.Button("Begin Conversation")

    conversation_output = gr.Textbox(label="Conversation Output", lines=20)
    debugging_output = gr.Textbox(label="Debugging Output", lines=20, visible=debugme)
    report_output = gr.Textbox(label="Report", lines=20)  # New output component

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
        inputs=[
            setup_info_input,
            council_question_input,
            topology_dropdown,
            agents_dataframe,
            moderator_prompt_textbox,
            max_iterations_input,
            suppress_webpage_popups,
            max_image_per_webpage,
            max_total_image_interpretations,
            suppress_research,
            prepare_report,
            report_instructions
        ],
        outputs=[conversation_output, debugging_output, report_output]
    )

demo.launch(share=False)
