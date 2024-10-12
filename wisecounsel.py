from typing import Annotated, Sequence, TypedDict, Union, List
from langgraph.graph import Graph, END
import operator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import gradio as gr
import os
import keys
import json


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
numAgents = 2

# Define the state
class AgentState(TypedDict):
    conversation_messages: List[Union[SystemMessage, HumanMessage, AIMessage]]
    setup_messages: List[Union[SystemMessage, HumanMessage, AIMessage]]
    current_agent: str
    nextAgent: str
    agentSysPrompt: str
    currentInstructions: str

# Initialize LLMs
openai_llm = ChatOpenAI(model="gpt-4o", temperature=0)
claude_llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0)

# Feedback Agent Class
class FeedbackAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_message = ""

    def set_system_message(self, message):
        self.system_message = message

    def generate_response(self, state: AgentState) -> AgentState:
        # Always begin with the system message before any other messages
        messages = [SystemMessage(content=self.system_message)] + state['conversation_messages']
        response = self.llm.invoke(messages)
        return {
            "conversation_messages": [*state['conversation_messages'], response],
            "setup_messages": state['setup_messages'],
            "current_agent": "presenter",
            "nextAgent": "presenter",
            "currentInstructions": state['currentInstructions']
        }

# Presenter Agent Class
class PresenterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.future_prompt = ""

    def generate_initial_prompts(self, instruction, agent_names):
        prompt = (
            f"You are Moderator, a moderator for a discussion based on the following instruction: '{instruction}'. "
            f"You have available to you feedback agents: {', '.join(agent_names)} and no one else"
            "Please provide a JSON response that contains:\n"
            "1. 'future_instructions': A future prompt to Moderator with instructions on how you will moderate the discussion including: agent names at your disposal (do not ask anyone else for feedback but the agents), when to stop the conversation, how to guide the conversation forward by giving agents instructions and appropriate background context to answer your instructions, and how to order feedback agent participation. It should ask Moderator return ONLY JSON response with 'next_agent', which specifies the name of the next agent to transfer control to or 'END', and 'agent_instruction', which contains Moderator instruction to the feedback agent along with relevant context for feedback agent to answer. Reminder this prompt must instruct moderator to return ONLY JSON.\n"
            "2. 'feedback_prompts': A list containing a system prompt for each feedback agent, defining their role in the discussion that will be used throughout the conversation.\n\n"
            "Please return the JSON without any additional text or formatting."
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response

    def generate_response(self, state: AgentState) -> AgentState:
        if len(state['setup_messages']) == 2:  # Initial prompt
            agent_names = [f"feedback_{i}" for i in range(numAgents)]
            response = self.generate_initial_prompts(state['setup_messages'][-1].content, agent_names)
            try:
                # Clean response content by removing code fences
                cleaned_content = clean_response(response.content)
                response_data = json.loads(cleaned_content)
                self.set_future_prompt(response_data['future_instructions'])
                feedback_prompts = response_data['feedback_prompts']
                for i, prompt in enumerate(feedback_prompts[:numAgents]):
                    feedback_agents[i].set_system_message(prompt)
                state['currentInstructions'] = response_data['future_instructions']
                # Append initial response to setup messages
                state['setup_messages'].append(AIMessage(content=response.content))
                # Ensure presenter runs again after initial setup
                state['nextAgent'] = "presenter"
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Failed to parse initial prompts: {e}")
        else:
            # For non-initial prompts, use future instructions as system message
            if len(state['setup_messages'])==3: # first non creation prompt
                messages = [SystemMessage(content=self.future_prompt)] + state['setup_messages'] + state['conversation_messages']
            else:    
                messages = [SystemMessage(content=self.future_prompt)] + state['conversation_messages']
            response = self.llm.invoke(messages)
            try:
                # Clean response content by removing code fences
                cleaned_content = clean_response(response.content)
                response_data = json.loads(cleaned_content)
                
                state['nextAgent'] = response_data['next_agent']
                state['currentInstructions'] = response_data['next_agent']+ ","+ response_data['agent_instruction']
                # Append only the agent_instruction to the conversation messages
                state['conversation_messages'].append(HumanMessage(content=state['currentInstructions']))
                if state['nextAgent'] == 'END':
                    return {
                        "conversation_messages": state['conversation_messages'],
                        "setup_messages": state['setup_messages'],
                        "current_agent": "END",
                        "nextAgent": "END",
                        "currentInstructions": state['currentInstructions']
                    }
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Failed to parse response for future instructions: {e}")
        return {
            "conversation_messages": state['conversation_messages'],
            "setup_messages": state['setup_messages'],
            "current_agent": "presenter",  # Always return to presenter after initial setup
            "nextAgent": state['nextAgent'],
            "currentInstructions": state['currentInstructions']
        }

    def set_future_prompt(self, prompt):
        self.future_prompt = prompt

# Initialize agents
presenter_agent = PresenterAgent(openai_llm)
feedback_agents = [
    FeedbackAgent(claude_llm),
    FeedbackAgent(openai_llm)
]

# Define the router function
def should_continue(state: AgentState) -> str:
    # Example condition: continue if less than 10 messages and conversation is not set to end

    return state['nextAgent']

    

workflow = Graph()

# Add nodes
workflow.add_node("presenter", presenter_agent.generate_response)
for i in range(numAgents):
    workflow.add_node(f"feedback_{i}", feedback_agents[i].generate_response)

# Add conditional edges
workflow.add_conditional_edges(
    "presenter",
    should_continue
)
for i in range(numAgents):
    workflow.add_conditional_edges(
        f"feedback_{i}",
        should_continue)
    

# Set the entrypoint
workflow.set_entry_point("presenter")

# Connect everything together
chain = workflow.compile()

# Gradio interface function
def run_wise_counsel(instruction, max_iterations=5):
    setup_messages = [SystemMessage(content="You are a wise counsel discussing important topics."), 
                      HumanMessage(content=instruction)]
    state = {"conversation_messages": [], "setup_messages": setup_messages, "current_agent": "presenter", "nextAgent": "presenter", "agentSysPrompt": "", "currentInstructions": ""}
    
    output = format_messages(state["setup_messages"])
    yield output

    for i, s in enumerate(chain.stream(state)):
        if i >= max_iterations - 1:
            break
        
        # Debug: Print the entire state
        print(f"Debug - State: {s}")
        
        # Check if the current agent's state is in the output
        current_agent = list(s.keys())[0]  # Get the current agent (presenter or feedback)
        if 'conversation_messages' in s[current_agent]:
            new_messages = s[current_agent]['conversation_messages']
            # Only add new messages that aren't already in the output
            new_content = format_messages([msg for msg in new_messages if msg.content not in output])
            if new_content:
                output += "\n\n" + new_content
        else:
            output += f"\n\nError: Unable to find messages for {current_agent}"
        
        yield output

def format_messages(messages):
    return "\n\n".join([f"{m.__class__.__name__}: {m.content}" for m in messages])

iface = gr.Interface(
    fn=run_wise_counsel,
    inputs=[
        gr.Textbox(label="Instruction for Presenter"),
        gr.Slider(minimum=1, maximum=20, step=1, value=10, label="Max Iterations")
    ],
    outputs=gr.Textbox(label="Wise Counsel Output"),
    live=False
)

# Launch the interface
iface.launch(share=False)