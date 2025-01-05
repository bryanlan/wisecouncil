
**README.md**

# Wise Council: A Multi-Agent Conversational System with Tool Usage

This repository contains a multi-agent conversational framework built on top of several key components:

-   **LLMs** (via `llms.py`)
-   **Agent classes** (via `agents.py`)
-   **Tools** (e.g., `ResearchTool` and `RedditSentimentTool`)
-   **A Gradio-based UI** (via `main.py`)

The project showcases how different agents (moderator, feedback agents, setup agent) can coordinate, use external tools (like Google search and Reddit sentiment analysis), and maintain conversation state. It also includes logic to generate a final conversation transcript and an optional research report.

----------

## Features at a Glance

1.  **Multiple Agents**
    
    -   **FeedbackAgent**: Listens to user input and conversation context, decides whether to invoke tools, and composes replies.
    -   **ModeratorAgent**: Oversees conversation flow, can direct the conversation to other agents, or decide if the conversation ends.
    -   **SetupAgent**: Dynamically configures the system (agents, roles, LLM assignments) based on user-provided setup information.
2.  **Tools**
    
    -   **ResearchTool** (`tools.py`): Integrates a custom Google search flow (via Selenium), extracts textual information from websites, cleans and summarizes the content, and returns consolidated results to the requesting agent.
    -   **RedditSentimentTool** (`sentiment.py`): Uses the `redditlib.py` module to query Reddit for sentiment/opinion analysis and gather relevant images; includes optional image interpretation using the cheap LLM.
3.  **LLM Abstractions** (`llms.py`)
    
    -   Wraps different LLM services (OpenAI GPT-4, Anthropic Claude, Google Gemini, etc.) in a retry mechanism (`RetryLLM`).
    -   Exposes a “cheap” LLM (for short classification or decision tasks) and a “state-of-the-art” LLM (for more comprehensive tasks).
4.  **Conversation State** (`state.py`)
    
    -   Maintains conversation messages and metadata (e.g., next agent to speak, maximum tool invocations, debugging info).
5.  **Conversation Flow** (`main.py`)
    
    -   Implements a Gradio interface to manage the multi-agent conversation.
    -   Allows user to specify setup information, question for the council, and advanced parameters like whether to prepare a final report or manually override the moderator.
    -   Streams conversation output in real-time and shows a debugging log if desired.

----------

## Project Structure

Below is a high-level overview of the primary files and their roles:

```
.
├── agents.py         # Core agent classes (BaseAgent, FeedbackAgent, ModeratorAgent, SetupAgent)
├── main.py           # Gradio-based app that orchestrates user interaction, conversation loop
├── llms.py           # Wrappers around various LLM backends, each wrapped with a retry mechanism
├── redditlib.py      # Helper module to query the Reddit API (via PRAW), fetch posts/comments
├── sentiment.py      # Defines a RedditSentimentTool using redditlib for sentiment/opinion analysis
├── state.py          # AgentState typed dictionary to track conversation context
├── tools.py          # Tool base class + ResearchTool to query and scrape web results
├── utils.py          # Utility functions for text cleaning, JSON parsing, formatting
└── requirements.txt  # (Optional) Python dependencies

```

----------

## Quick Start

1.  **Install Dependencies**  
    Make sure you install all required packages. The project uses (among others):
    
    -   `gradio`
    -   `requests`
    -   `praw`
    -   `selenium`
    -   `webdriver_manager`
    -   `beautifulsoup4`
    -   `html2text`
    -   `justext`
    -   `langchain_core` (or your local variant)
    -   `tenacity`
    -   And more...
    
    ```bash
    pip install -r requirements.txt
    
    ```
    
2.  **Set Up Credentials**
    
    -   **Google**: A valid [Google Custom Search API key and Search Engine ID](https://developers.google.com/custom-search/v1/overview) in `config.keys.GOOGLE_API_KEY` and `config.keys.GOOGLE_SEARCH_ENGINE_ID`.
    -   **Reddit**: A valid Reddit API client ID, secret, and user agent in `keys.py` (or similar) for the `redditlib.py` module.
    -   **OpenAI / Anthropic / Others**: If you plan to use them, ensure you have the appropriate environment variables / keys set.
3.  **Launch the Application**
    
    ```bash
    python main.py
    
    ```
    
    This will spin up a Gradio interface, typically available at `http://127.0.0.1:7860`.
    
4.  **Interact with the Gradio UI**
    
    -   **Setup Information**: Provide an overall scenario or goal for your conversation.
    -   **Question for the Council**: The main user query or topic for agents to discuss.
    -   **Agent Control Flow**: Select how agents should be invoked (e.g., round robin, last decides next, moderator-based, etc.).
    -   **Generate Setup**: This step calls the `SetupAgent` to propose a set of agents, roles, and LLM assignments.
    -   **Begin Conversation**: Kicks off the conversation loop. If “human moderator override” is ON, you can modify the moderator’s instructions after each iteration to direct the conversation.
  
![image](https://github.com/user-attachments/assets/e9fad3d6-cd7c-4b9d-b5ef-090ade163321)


----------

## How It Works

1.  **Setup Phase**  
    The `SetupAgent` takes user instructions from the Gradio UI and automatically proposes a JSON structure specifying which agents to create, their roles, and which LLMs they use.
    
2.  **Main Conversation Loop**
    
    -   The conversation loop is orchestrated in `main.py`.
    -   The `run_conversation_iter` function calls whichever agent is next. Each agent can optionally invoke tools (search, Reddit sentiment) or produce a direct response.
    -   A “moderator” agent can redirect the flow or decide when to end (`END`).
3.  **Tool Invocations**
    
    -   Agents attempt to determine if a tool is needed. If needed, they pass the input to the corresponding tool method.
    -   Each tool (e.g., `ResearchTool`, `RedditSentimentTool`) returns summarized or raw results.
    -   The agent incorporates these results back into the conversation context.
4.  **Output & Debugging**
    
    -   Messages are saved in the conversation state.
    -   A final “report” can optionally be generated from the entire conversation transcript if the user selects **Prepare Report**.
    -   **Debugging Output** provides detailed logs of LLM calls, partial responses, and chain-of-thought (if enabled).

----------

## Customizing

-   **LLM Providers**: Modify or add new LLM providers in `llms.py`.
-   **New Tools**: Extend `Tool` in `tools.py` or create a separate file. Overwrite the `invoke` method to implement custom logic.
-   **Extended Agents**: Subclass `BaseAgent` or `FeedbackAgent` if you need more specialized agent behaviors.

----------

## Security & Rate Limits

-   **OpenAI/Anthropic** usage may incur costs and rate limits. Monitor your usage and handle errors with caution.
-   **Reddit** API usage is subject to [Reddit’s API rules](https://www.redditinc.com/policies/developer-terms). Provide a descriptive user agent and handle rate limits.
-   **Selenium-based scraping** must follow site-specific terms of use and robots.txt rules.

----------

## Contributing

1.  **Fork** this repository.
2.  **Create** a feature branch.
3.  **Commit** your changes with clear messages.
4.  **Push** to your fork and open a Pull Request.

----------

## License

This project is distributed under a license of your choice (add your LICENSE file accordingly). By using the code in this repository, you acknowledge and agree to the license terms.

----------

**Thank you for using Wise Council!**  
For inquiries, improvements, or feedback, please open an issue or a pull request. Enjoy building multi-agent systems that can research, reason, and converse with external data!
