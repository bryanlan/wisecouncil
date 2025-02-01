# llms.py

from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
from config import GEMINI_KEY, DEEPSEEK_API_KEY
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class SystemMessageHandlingLLM:
    """Wrapper for LLMs that handles system messages by converting them to user messages when needed."""
    def __init__(self, llm, supports_system_messages: bool = True):
        self.llm = llm
        self.supports_system_messages = supports_system_messages

    def _convert_messages(self, messages):
        """Convert system messages to user messages if the model doesn't support them."""
        if self.supports_system_messages:
            return messages

        converted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Convert system message to user message with a prefix
                converted_messages.append(HumanMessage(content=f"System instruction: {msg.content}"))
            else:
                converted_messages.append(msg)
        return converted_messages

    def _convert_dict_messages(self, messages):
        """Convert dict format messages to Message objects and handle system messages."""
        converted = []
        for msg in messages:
            if msg["role"] == "system":
                if self.supports_system_messages:
                    converted.append(SystemMessage(content=msg["content"]))
                else:
                    converted.append(HumanMessage(content=f"System instruction: {msg['content']}"))
            elif msg["role"] == "user":
                converted.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                converted.append(AIMessage(content=msg["content"]))
        return converted

    def invoke(self, *args, **kwargs):
        """Invoke the LLM with message conversion if needed."""
        # Handle both direct messages and input=messages formats
        if args and isinstance(args[0], (list, tuple)):
            messages = args[0]
            if isinstance(messages[0], dict):
                messages = self._convert_dict_messages(messages)
            else:
                messages = self._convert_messages(messages)
            return self.llm.invoke(messages, **kwargs)
        elif "input" in kwargs and isinstance(kwargs["input"], (list, tuple)):
            messages = kwargs["input"]
            if isinstance(messages[0], dict):
                messages = self._convert_dict_messages(messages)
            else:
                messages = self._convert_messages(messages)
            kwargs["input"] = messages
            return self.llm.invoke(**kwargs)
        return self.llm.invoke(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.llm, name)

class RetryLLM:
    def __init__(self, llm, max_attempts: int = 5, min_wait: float = 2, max_wait: float = 20):
        self.llm = llm
        self.max_attempts = max_attempts
        self.min_wait = min_wait
        self.max_wait = max_wait

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        reraise=True
    )
    def invoke(self, *args, **kwargs):
        try:
            return self.llm.invoke(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"Rate limit hit, retrying with exponential backoff...")
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        reraise=True
    )
    def batch(self, *args, **kwargs):
        try:
            return self.llm.batch(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"Rate limit hit, retrying with exponential backoff...")
            raise

    def __getattr__(self, name):
        return getattr(self.llm, name)

def create_llm(model_type: str, temperature: float = 0) -> RetryLLM:
    """Create an LLM with the specified model type and temperature"""
    # Clamp temperature between 0 and 2
    temperature = max(0, min(2, temperature))
    
    if model_type == "openai_gpt4o":
        return RetryLLM(SystemMessageHandlingLLM(ChatOpenAI(model="gpt-4o", temperature=temperature)))
    elif model_type == "anthropic_claude":
        return RetryLLM(SystemMessageHandlingLLM(ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=max(0, min(1, temperature)))))
    elif model_type == "google_gemini":
        return RetryLLM(SystemMessageHandlingLLM(ChatGoogleGenerativeAI(model="gemini-exp-1206", temperature=max(0, min(1, temperature)), api_key=GEMINI_KEY)))
    elif model_type == "xai_grok":
        return RetryLLM(SystemMessageHandlingLLM(ChatXAI(model="grok-beta", temperature=temperature)))
    elif model_type == "deepseek_chat":
        return RetryLLM(SystemMessageHandlingLLM(ChatOpenAI(model='deepseek-chat', temperature=temperature, openai_api_key=DEEPSEEK_API_KEY, openai_api_base='https://api.deepseek.com', max_tokens=8096)))
    elif model_type == "openai_gpt-o1":
        # o1-preview doesn't support system messages and only supports temperature=1
        return RetryLLM(SystemMessageHandlingLLM(
            ChatOpenAI(model="o1-preview", temperature=1),
            supports_system_messages=False
        ))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Initialize default LLMs with temperature=0
openai_llm = create_llm("openai_gpt4o")
openai_llm_reasoning = create_llm("openai_gpt-o1")
openai_llm_mini = RetryLLM(SystemMessageHandlingLLM(ChatOpenAI(model="gpt-4o-mini", temperature=0)))
claude_llm = create_llm("anthropic_claude")
gemini_llm = create_llm("google_gemini")
grok_llm = create_llm("xai_grok")
deepseek_llm = create_llm("deepseek_chat")
gemini_llm_flash = RetryLLM(SystemMessageHandlingLLM(ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, api_key=GEMINI_KEY)))

# Add this after the other constants
cheap_llm = deepseek_llm # Can be changed to other low-cost LLMs as needed
sota_llm = openai_llm



# Available LLMs with descriptions
available_llms = [
    {
        "name": "openai_gpt4o",
        "description": "OpenAI's GPT-4 model, a best-in-class foundation model capable of general use.",
        "llm": openai_llm
    },
    {
        "name": "anthropic_claude",
        "description": "Anthropic's Claude model, a best-in-class foundation model capable of general use.",
        "llm": claude_llm
    },
     {
        "name": "google_gemini",
        "description": "Google's Gemini model, a best-in-class foundation model capable of general use.",
        "llm": gemini_llm
     },
    {
        "name": "xai_grok",
        "description": "xAI's Grok model, an AI chatbot integrated with X. Best for espousing controversial viewpoints.",
        "llm": grok_llm
    },
    {
        "name": "deepseek_chat",
        "description": "DeepSeek's Chat model, a best-in-class foundation model capable of general use.",
        "llm": deepseek_llm
    },
    {
        "name": "openai_gpt-o1",
        "description": "Open AI's o1 reasoning model.",
        "llm": openai_llm_reasoning
    }
]
if __name__ == "__main__":
    # Select a model to test (e.g., OpenAI GPT-4)
    model_to_test = deepseek_llm

    # Define a test prompt using the expected input format (list of messages)
    test_input = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the three laws of robotics?"}
    ]

    try:
        # Invoke the model with the test input
        response = model_to_test.invoke(input=test_input)
        print("Response from the model:")
        print(response)
    except Exception as e:
        print("An error occurred while testing the model:")
        print(e)
