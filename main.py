# main.py

import gradio as gr
from typing import List, Dict, Any, Tuple
import time
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from llms import available_llms, sota_llm, create_llm
from config import debugme, TOOLPREFIX, moderatorPromptEnd, PROMPT_LENGTH_THRESHOLD
from utils import clean_response, format_messages
from agents import SetupAgent, FeedbackAgent, ModeratorAgent
from tools import ResearchTool
from sentiment import RedditSentimentTool  # NEW import for the Reddit sentiment tool
from state import AgentState
import copy

###############################################################################
# SETUP AGENT
###############################################################################

setup_agent = SetupAgent(sota_llm)

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def setup_conversation(setup_info: str):
    """
    Generates the conversation setup and handles UI state clearing.
    Returns tuple of (agents_df, moderator_prompt, conversation_output, debug_output, 
    report_output, conversation_store, radio_update, moderator_response).
    """
    try:
        setup_data = setup_agent.generate_setup(setup_info)
    except ValueError as e:
        return (
            [], 
            "",
            "Error in setup generation: " + str(e),  # conversation output
            "",  # debug output
            "",  # report output
            None,  # conversation store
            gr.update(value=None, choices=[]),  # radio buttons
            gr.update(value="")  # moderator response
        )

    agents = setup_data.get('agents', [])
    moderator_prompt = setup_data.get('moderator_prompt', '')

    if moderator_prompt != '':
        moderator_prompt += moderatorPromptEnd

    # Convert to list-of-lists for the Gradio dataframe, adding temperature=0
    agents_df = [
        [agent['name'], agent['type'], agent['role'], 0.0] 
        for agent in agents
    ]

    return (
        agents_df,
        moderator_prompt,
        "",  # Clear conversation output
        "",  # Clear debugging output
        "",  # Clear report output
        None,  # Clear conversation store state
        gr.update(value=None, choices=[]),  # Clear radio buttons
        gr.update(value="")  # Clear moderator response
    )

def collect_debugging_text(state: AgentState) -> str:
    """
    Aggregates debugging_info from the state into a single text string.
    """
    output_debug = ""
    if debugme == 1 and "debugging_info" in state:
        for debug_entry in state["debugging_info"]:
            agent_name = debug_entry.get('agent_name', 'Unknown')
            messages_sent = debug_entry.get('messages_sent', [])
            response = debug_entry.get('response', '')
            next_agent = debug_entry.get('next_agent', 'Not specified')
            output_debug += f"\n\nAgent: {agent_name}\nMessages Sent:\n"
            output_debug += format_messages(messages_sent, False)
            output_debug += f"\n\nResponse:\n{response}"
            output_debug += f"\nNext Agent: {next_agent}"
    return output_debug

def produce_report_if_requested(
    state: AgentState, 
    prepare_report: bool, 
    reporterPrompt: str, 
    council_question: str
) -> str:
    """
    Optionally produce a final report using sota_llm, if requested.
    """
    if not prepare_report:
        return ""
    conversation_transcript = format_messages(
        state["conversation_messages"], 
        suppressToolMsg=False
    )
    # Replace placeholder
    report_prompt = reporterPrompt.replace("[council question]", council_question)
    report_prompt = (
        f"{report_prompt}\n\n"
        f"Conversation and Tool Outputs:\n{conversation_transcript}"
    )
    response = sota_llm.invoke([HumanMessage(content=report_prompt)])
    return response.content.strip()

###############################################################################
# ONE-STEP CHAIN EXECUTION
###############################################################################

def run_conversation_iter(conversation_store: dict) -> dict:
    """
    Runs exactly ONE iteration (one step) of the conversation.
    """
    # Make a deep copy of the state to ensure we don't accidentally modify the original
    state = copy.deepcopy(conversation_store["state"])

    if state["nextAgent"] == "END":
        return conversation_store
    if conversation_store["iteration_count"] >= conversation_store["max_iterations"]:
        state["nextAgent"] = "END"
        conversation_store["state"] = state
        return conversation_store

    conversation_store["iteration_count"] += 1

    # Get the current agent based on nextAgent
    current_agent = None
    if state["nextAgent"] in conversation_store["agent_objects"]:
        current_agent = conversation_store["agent_objects"][state["nextAgent"]]
    elif state["nextAgent"] == state["moderatorName"]:
        current_agent = conversation_store["moderator_agent"]
    
    if current_agent:
        # Check if we've hit the tool invocation limit
        if state.get('tool_invocation_count', 0) >= state.get('max_tool_invocations', float('inf')):
            # Remove tools from the agent temporarily
            current_agent.available_tools = []
        
        # Execute the current agent's response generation
        updated_state = current_agent.generate_response(state)
        
        # Restore tools to the agent if we removed them
        if not current_agent.available_tools:
            current_agent.available_tools = conversation_store.get("original_tools", [])
            
        # Ensure we preserve any new messages
        conversation_store["state"] = updated_state
    else:
        state["nextAgent"] = "END"
        conversation_store["state"] = state

    return conversation_store


def apply_moderator_override(
    conversation_store: dict,
    user_moderator_text: str,
    user_next_agent: str
) -> dict:
    """
    Merges the human override text & next-agent choice into the conversation state.
    The conversation must be at the moderator step to reach this function.
    """
    state = conversation_store["state"]
    
    # Replace the last message with the override text
    if state["conversation_messages"]:
        state["conversation_messages"][-1] = HumanMessage(content="Moderator: " + user_moderator_text)
    
    # Force next agent if user selected one, otherwise end conversation
    state["nextAgent"] = user_next_agent if user_next_agent else "END"

    conversation_store["state"] = state
    return conversation_store

###############################################################################
# MAIN ENTRYPOINT
###############################################################################

def run_conversation_init(
    setup_info: str,
    council_question: str,
    topology_type: str,
    agents_list: List[Dict[str, Any]],
    moderator_prompt: str,
    max_iterations: int,
    suppress_webpage_popups: bool,
    disable_tools: bool,
    max_image_per_webpage: int,
    max_total_image_interpretations: int,
    suppressResearch: bool,
    prepare_report: bool,
    reporterPrompt: str,
    human_moderator_override: bool,
    max_tool_invocations: int
) -> Tuple[str, str, str, dict]:
    """
    Initializes the conversation:
      - Builds the agents & state.
      - If override=OFF, runs the entire conversation automatically to completion
        or until max_iterations is reached.
      - If override=ON, runs exactly one iteration and returns partial results.
    Returns (conversation_output, debugging_output, report_output, conversation_store).
    """
    # Build agent objects
    agent_objects = {}
    moderator_agent = None
    moderatorName = ""

    available_tools = []
    if not disable_tools:
        # Add the standard research tool
        available_tools.append(
            ResearchTool(
                dismissPopups=suppress_webpage_popups,
                max_interpretations_per_page=max_image_per_webpage,
                max_total_image_interpretations=max_total_image_interpretations
            )
        )
        # Add the new Reddit sentiment tool
        available_tools.append(
            RedditSentimentTool()
        )

    for agent_info in agents_list:
        name = agent_info['name']
        llm_type = agent_info['type']
        role = agent_info['role']
        temperature = agent_info.get('temperature', 0.0)  # Get temperature with default 0
        isModerator = ('moderator' in name.lower())

        # Create the LLM with the specified temperature
        llm = create_llm(llm_type, temperature)

        if topology_type in ['moderator_discretionary', 'moderated_round_robin'] and isModerator:
            moderator_agent = ModeratorAgent(llm, available_tools)
            moderator_agent.set_setup({
                'topology_type': topology_type,
                'agents': agents_list,
                'moderator_prompt': moderator_prompt
            })
            moderatorName = name
        else:
            agent_objects[name] = FeedbackAgent(
                name=name,
                llm=llm,
                role=role,
                topology=topology_type,
                agent_info=agents_list,
                available_tools=available_tools
            )

    # Ensure a moderator is present if needed
    if topology_type in ['moderator_discretionary', 'moderated_round_robin'] and moderator_agent is None:
        return ("Error: No moderator found in agents list.", "", "", {})

    # Find moderator name from agents list
    moderatorName = ""
    for agent_info in agents_list:
        name = agent_info['name']
        if 'moderator' in name.lower():
            moderatorName = name
            break

    # Initialize state
    state: AgentState = {
        "conversation_messages": [HumanMessage(content=council_question)],
        "setup_messages": [],
        "current_agent": "",
        "nextAgent": "",
        "agentSysPrompt": "",
        "agent_order": [],
        "topology": topology_type,
        "agents": agents_list,
        "control_flow_index": 0,
        "debugging_info": [],
        "moderatorName": moderatorName,
        "tool_invocation_count": 0,
        "max_tool_invocations": max_tool_invocations
    }

    # Set initial nextAgent based on topology
    if topology_type in ['moderator_discretionary', 'moderated_round_robin']:
        state["nextAgent"] = moderatorName
    elif topology_type == 'round_robin' and agents_list:
        state["nextAgent"] = agents_list[0]['name']
    elif topology_type == 'last_decides_next' and agents_list:
        state["nextAgent"] = agents_list[0]['name']

    # If debug is on, reset debug.txt
    if debugme == 1:
        with open('debug.txt', 'w', encoding='utf-8') as f:
            f.write(f"=== New Debug Session {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    # Wrap state & agents
    conversation_store = {
        "state": state,
        "agent_objects": agent_objects,
        "moderator_agent": moderator_agent,
        "suppressResearch": suppressResearch,
        "max_iterations": max_iterations,
        "iteration_count": 0,
        "prepare_report": prepare_report,
        "reporterPrompt": reporterPrompt,
        "override_active": human_moderator_override,
        "tool_invocation_count": 0,
        "max_tool_invocations": max_tool_invocations
    }

    # If override=OFF, run entire conversation automatically
    if not human_moderator_override:
        while conversation_store["iteration_count"] < max_iterations:
            conversation_store = run_conversation_iter(conversation_store)
            if conversation_store["state"]["nextAgent"] == "END":
                break

        final_state = conversation_store["state"]
        conversation_output = format_messages(final_state["conversation_messages"], suppressResearch)
        debugging_output = collect_debugging_text(final_state)
        report_output = produce_report_if_requested(
            final_state, 
            prepare_report, 
            reporterPrompt, 
            council_question
        )
        return conversation_output, debugging_output, report_output, conversation_store
    else:
        # If override=ON, run exactly one iteration and return partial results
        conversation_store = run_conversation_iter(conversation_store)
        partial_state = conversation_store["state"]
        conversation_output = format_messages(partial_state["conversation_messages"], suppressResearch)
        debugging_output = collect_debugging_text(partial_state)
        report_output = ""
        return conversation_output, debugging_output, report_output, conversation_store


###############################################################################
# GRADIO APP
###############################################################################

with gr.Blocks(css="""
    .debugging-output > label > textarea {
        overflow-y: auto !important;
        max-height: 500px !important;
        height: 500px !important;
    }
    .pretty_scrollbar::-webkit-scrollbar {
        width: 10px;
    }
    .pretty_scrollbar::-webkit-scrollbar-track {
        background: transparent;
    }
    .pretty_scrollbar::-webkit-scrollbar-thumb,
    .pretty_scrollbar::-webkit-scrollbar-thumb:hover {
        background: #c5c5d2;
        border-radius: 10px;
    }
    .dark .pretty_scrollbar::-webkit-scrollbar-thumb,
    .dark .pretty_scrollbar::-webkit-scrollbar-thumb:hover {
        background: #374151;
        border-radius: 10px;
    }
    .pretty_scrollbar::-webkit-resizer {
        background: #c5c5d2;
    }
    .dark .pretty_scrollbar::-webkit-resizer {
        background: #374151;
    }
""") as demo:
    gr.Markdown("## Wise Council")

    ############################################################################
    # Row: Setup Info
    ############################################################################
    with gr.Row():
        setup_info_input = gr.Textbox(label="Setup Information", lines=2)
        council_question_input = gr.Textbox(label="Question for the Council", lines=2)
        report_instructions = gr.Textbox(
            label="Prompt for reporter", 
            value=(
                "Using the following conversation and tool outputs, create a research report "
                "designed to answer the question: [council question].\n"
                "The report should be very detailed and scientific, presenting uncertainty in "
                "a probabilistic fashion, and if necessary, contain subjects for further "
                "research or next steps, as appropriate."
            ), 
            lines=2
        )

    ############################################################################
    # Row: Checkboxes, including Human moderator override
    ############################################################################
    with gr.Row():
        suppress_webpage_popups = gr.Checkbox(label="Suppress Webpage Popups (very slow)", value=False)
        disable_tools = gr.Checkbox(label="Disable Tool Usage", value=False)
        max_image_per_webpage = gr.Number(label="Max Image Interpretations per Webpage", value=4)
        max_total_image_interpretations = gr.Number(label="Max Total Image Interpretations", value=10)
        suppress_research = gr.Checkbox(label="Suppress Research in Chat", value=False)
        prepare_report = gr.Checkbox(label="Prepare Report", value=False)
        human_moderator_override = gr.Checkbox(label="Human moderator override", value=False)

    ############################################################################
    # Setup, Agents, Moderator Prompt
    ############################################################################
    with gr.Row():
        topology_dropdown = gr.Dropdown(
            choices=[
                'round_robin', 
                'last_decides_next', 
                'moderator_discretionary', 
                'moderated_round_robin'
            ],
            label="Agent Control Flow"
        )
    setup_button = gr.Button("Generate Setup")
   

    with gr.Row():
        agents_dataframe = gr.Dataframe(
            headers=['name', 'type', 'role', 'temperature'],
            datatype=['str', 'str', 'str', 'number'],
            interactive=True,
            label="Agents (Name, Type, Role, Temperature)",
            value=[["", "", "", 0.0]]
        )

    # Add new table for Available LLMs
    with gr.Row():
        llms_dataframe = gr.Dataframe(
            headers=['model type', 'model description'],
            datatype=['str', 'str'],
            interactive=False,
            label="Available LLMs",
            value=[[llm['name'], llm.get('description', 'No description available')] for llm in available_llms]
        )

    with gr.Row():
        moderator_prompt_textbox = gr.Textbox(label="Moderator Prompt", lines=5)


    max_iterations_input = gr.Slider(
        minimum=1,
        maximum=40,
        step=1,
        value=5,
        label="Max Iterations"
    )
    max_tool_invocations = gr.Slider(
        minimum=0,
        maximum=20,
        step=1,
        value=3,
        label="Max Tool Invocations"
    )
    run_button = gr.Button("Begin Conversation")

    ############################################################################
    # Conversation Output
    ############################################################################
    conversation_output = gr.Textbox(label="Conversation Output", lines=20)

    ############################################################################
    # Moderator Override UI (below conversation output)
    ############################################################################
    human_moderator_response = gr.Textbox(
        label="Human moderator's response",
        lines=5,
        visible=False
    )
    which_agent_next = gr.Radio(
        choices=[], 
        label="Which agent should respond next?",
        value=None,
        visible=False
    )
    apply_override_button = gr.Button("Apply Override & Continue", visible=False)

    ############################################################################
    # Debugging & Report
    ############################################################################
    debugging_output = gr.Textbox(
        label="Debugging Output",
        lines=20,
        visible=debugme,
        elem_classes=["debugging-output", "pretty_scrollbar"]
    )
    report_output = gr.Textbox(label="Report", lines=20)

    ############################################################################
    # Keep a Gradio State for conversation storage
    ############################################################################
    conversation_store_state = gr.State()

    ############################################################################
    # Setup button logic: call setup_conversation
    ############################################################################
    setup_button.click(
        fn=setup_conversation,
        inputs=setup_info_input,
        outputs=[
            agents_dataframe,
            moderator_prompt_textbox,
            conversation_output,
            debugging_output,
            report_output,
            conversation_store_state,
            which_agent_next,
            human_moderator_response
        ]
    )


    ############################################################################
    # Toggle moderator override UI
    ############################################################################
    def toggle_override_ui(checked: bool):
        """ Show or hide the override text/radio/button. """
        if checked:
            return (
                gr.update(visible=True), 
                gr.update(visible=True),
                gr.update(visible=True)
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )

    human_moderator_override.change(
        fn=toggle_override_ui,
        inputs=human_moderator_override,
        outputs=[human_moderator_response, which_agent_next, apply_override_button]
    )

    ############################################################################
    # Begin Conversation
    ############################################################################
    def begin_conversation(
        setup_info,
        council_question,
        topology_type_val,
        agents_df_val,
        moderator_prompt_val,
        max_iterations_val,
        suppress_popups_val,
        disable_tools_val,
        max_img_per_page_val,
        max_total_img_val,
        suppress_research_val,
        prepare_report_val,
        report_instructions_val,
        override_val,
        max_tool_invocations_val
    ):
        """
        Calls run_conversation_init. If override=OFF, it returns the final 
        conversation. If override=ON, it returns a partial after 1 iteration.
        Then we set the which_agent_next's choices to the full agent names (except Moderator).
        """
        # Convert agents DataFrame to list of dicts with temperature
        agent_configs = []
        if isinstance(agents_df_val, list):  # If it's a list of lists
            for row in agents_df_val:
                if len(row) >= 4:  # Ensure row has name, type, role, temperature
                    agent_configs.append({
                        'name': row[0].strip(),
                        'type': row[1].strip(),
                        'role': row[2].strip(),
                        'temperature': float(row[3]) if row[3] else 0.0
                    })
        else:  # If it's a DataFrame
            for _, row in agents_df_val.iterrows():
                agent_configs.append({
                    'name': row[0].strip(),
                    'type': row[1].strip(),
                    'role': row[2].strip(),
                    'temperature': float(row[3]) if row[3] else 0.0
                })

        conv_out, debug_out, rep_out, conv_store = run_conversation_init(
            setup_info,
            council_question,
            topology_type_val,
            agent_configs,  # Pass the enhanced agent configs
            moderator_prompt_val,
            max_iterations_val,
            suppress_popups_val,
            disable_tools_val,
            max_img_per_page_val,
            max_total_img_val,
            suppress_research_val,
            prepare_report_val,
            report_instructions_val,
            override_val,
            max_tool_invocations_val
        )

        # Build the list of agent names from the DataFrame
        agent_names = []
        if isinstance(agents_df_val, list):  # If it's a list of lists
            for row in agents_df_val:
                if len(row) >= 3:  # Ensure row has name, type, role
                    agent_names.append(row[0].strip())
        else:  # If it's a DataFrame
            for _, row in agents_df_val.iterrows():
                agent_names.append(row[0].strip())

        # Exclude the moderator from the radio list
        mod_name = ""
        for name in agent_names:
            if 'moderator' in name.lower():
                mod_name = name
                break
        agent_choices = [name for name in agent_names if name != mod_name]

        # Get the AI moderator's response and next agent selection
        ai_mod_response = ""
        ai_next_agent = None
        if conv_store and "state" in conv_store:
            state = conv_store["state"]
            if state["conversation_messages"]:
                last_msg = state["conversation_messages"][-1]
                if isinstance(last_msg, (AIMessage, HumanMessage)):
                    ai_mod_response = last_msg.content.replace("Moderator: ", "", 1)
                    ai_next_agent = state["nextAgent"]

        # Return partial or final conversation
        return (
            conv_out,
            debug_out,
            rep_out,
            conv_store,
            gr.update(choices=agent_choices, value=ai_next_agent),
            gr.update(value=ai_mod_response)  # Set default value for human_moderator_response
        )

    run_button.click(
        fn=begin_conversation,
        inputs=[
            setup_info_input,
            council_question_input,
            topology_dropdown,
            agents_dataframe,
            moderator_prompt_textbox,
            max_iterations_input,
            suppress_webpage_popups,
            disable_tools,
            max_image_per_webpage,
            max_total_image_interpretations,
            suppress_research,
            prepare_report,
            report_instructions,
            human_moderator_override,
            max_tool_invocations
        ],
        outputs=[
            conversation_output,
            debugging_output,
            report_output,
            conversation_store_state,
            which_agent_next,
            human_moderator_response
        ]
    )

    ############################################################################
    # Apply Override & Continue
    ############################################################################
    def apply_override_and_continue(
        conversation_store: dict, 
        user_moderator_text: str, 
        user_next_agent: str
    ):
        """
        1) Override the moderator text if we are at the moderator step
        2) Run exactly one iteration of the chain
        3) If we just ran a feedback agent, also run the moderator's response
        4) Return updated conversation, debugging, possibly final report if ended
        """
        # Check if next agent is selected when using override
        if not user_next_agent:
            return (
                gr.update(value="ERROR: You must select which agent should respond next before continuing."),
                gr.update(),  # Keep debugging output unchanged
                gr.update(),  # Keep report output unchanged
                conversation_store,  # Keep conversation store unchanged
                gr.update(),  # Keep radio buttons unchanged
                gr.update()  # Keep moderator response unchanged
            )

        # 1) apply override
        # Only prepend agent name if not already present
        if user_next_agent:
            text_lower = user_moderator_text.strip().lower()
            agent_lower = user_next_agent.lower()
            if not text_lower.startswith(agent_lower):
                user_moderator_text = f"{user_next_agent}, {user_moderator_text}"

        updated_store = apply_moderator_override(
            conversation_store, 
            user_moderator_text, 
            user_next_agent
        )

        # 2) run one iteration of feedback agent with the override moderator text
        updated_store = run_conversation_iter(updated_store)

        # 3) If we just ran a feedback agent and conversation isn't ended,
        # also run the moderator to get their response ready
        state = updated_store["state"]
        if (state["nextAgent"] != "END" and 
            updated_store["iteration_count"] < updated_store["max_iterations"]):
            
            # Run the moderator's iteration to get their response
            updated_store = run_conversation_iter(updated_store)
            
            # Get the AI moderator's response and next agent selection
            ai_mod_response = ""
            ai_next_agent = None
            if updated_store["state"]["conversation_messages"]:
                last_msg = updated_store["state"]["conversation_messages"][-1]
                if isinstance(last_msg, (AIMessage, HumanMessage)):
                    ai_mod_response = last_msg.content.replace("Moderator: ", "", 1)
                    ai_next_agent = updated_store["state"]["nextAgent"]
        else:
            ai_mod_response = ""
            ai_next_agent = state["nextAgent"]

        # Build conversation output
        state = updated_store["state"]
        conv_out = format_messages(state["conversation_messages"], updated_store["suppressResearch"])
        debug_out = collect_debugging_text(state)

        # If we ended or reached max
        report_out = ""
        if state["nextAgent"] == "END" or updated_store["iteration_count"] >= updated_store["max_iterations"]:
            # Possibly generate final report
            question_text = ""
            if state["conversation_messages"]:
                # The first message was the user question
                question_text = state["conversation_messages"][0].content
            report_out = produce_report_if_requested(
                state,
                updated_store["prepare_report"],
                updated_store["reporterPrompt"],
                question_text
            )

        # Get list of agent choices (excluding moderator)
        agent_choices = []
        mod_name = state["moderatorName"]
        for agent in updated_store.get("state", {}).get("agents", []):
            name = agent.get("name", "")
            if name and name != mod_name:
                agent_choices.append(name)

        # Update the radio buttons and text box with AI moderator's defaults
        radio_update = gr.update(choices=agent_choices, value=ai_next_agent)
        text_update = gr.update(value=ai_mod_response)

        return conv_out, debug_out, report_out, updated_store, radio_update, text_update

    apply_override_button.click(
        fn=apply_override_and_continue,
        inputs=[conversation_store_state, human_moderator_response, which_agent_next],
        outputs=[
            conversation_output, 
            debugging_output, 
            report_output, 
            conversation_store_state,
            which_agent_next,
            human_moderator_response
        ]
    )

demo.launch(share=False)

