# llms.py

from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
from config import GEMINI_KEY, DEEPSEEK_API_KEY

class RetryLLM:
    def __init__(self, llm, max_attempts: int = 3, min_wait: float = 1, max_wait: float = 10):
        self.llm = llm
        self.max_attempts = max_attempts
        self.min_wait = min_wait
        self.max_wait = max_wait

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    def invoke(self, *args, **kwargs):
        return self.llm.invoke(*args, **kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    def batch(self, *args, **kwargs):
        return self.llm.batch(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.llm, name)

def create_llm(model_type: str, temperature: float = 0) -> RetryLLM:
    """Create an LLM with the specified model type and temperature"""
    # Clamp temperature between 0 and 2
    temperature = max(0, min(2, temperature))
    
    if model_type == "openai_gpt4o":
        return RetryLLM(ChatOpenAI(model="gpt-4o", temperature=temperature))
    elif model_type == "anthropic_claude":
        return RetryLLM(ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=temperature))
    elif model_type == "google_gemini":
        return RetryLLM(ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=temperature, api_key=GEMINI_KEY))
    elif model_type == "xai_grok":
        return RetryLLM(ChatXAI(model="grok-beta", temperature=temperature))
    elif model_type == "deepseek_chat":
        return RetryLLM(ChatOpenAI(model='deepseek-chat', temperature=temperature, openai_api_key=DEEPSEEK_API_KEY, openai_api_base='https://api.deepseek.com', max_tokens=8096))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Initialize default LLMs with temperature=0
openai_llm = create_llm("openai_gpt4o")
openai_llm_mini = RetryLLM(ChatOpenAI(model="gpt-4o-mini", temperature=0))
claude_llm = create_llm("anthropic_claude")
gemini_llm = create_llm("google_gemini")
grok_llm = create_llm("xai_grok")
deepseek_llm = create_llm("deepseek_chat")
gemini_llm_flash = RetryLLM(ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, api_key=GEMINI_KEY))

# Add this after the other constants
cheap_llm = openai_llm_mini # Can be changed to other low-cost LLMs as needed
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
        "description": "Google's Gemini model, a state-of-the-art multimodal AI model.",
        "llm": gemini_llm
     },
    {
        "name": "xai_grok",
        "description": "xAI's Grok model, an AI chatbot integrated with X. Best for espousing controversial viewpoints.",
        "llm": grok_llm
    },
    {
        "name": "deepseek_chat",
        "description": "DeepSeek's Chat model, cost-effective foundational model.",
        "llm": deepseek_llm
    }
]
if __name__ == "__main__":
    # Select a model to test (e.g., OpenAI GPT-4)
    model_to_test = gemini_llm_flash

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
