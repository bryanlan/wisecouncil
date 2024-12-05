# llms.py

from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
from config import GEMINI_KEY

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

# Initialize your LLMs with retry wrapper
openai_llm = RetryLLM(ChatOpenAI(model="gpt-4o", temperature=0))
openai_llm_mini = RetryLLM(ChatOpenAI(model="gpt-4o-mini", temperature=0))
claude_llm = RetryLLM(ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0))
gemini_llm = RetryLLM(ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, api_key=GEMINI_KEY))
grok_llm = RetryLLM(ChatXAI(model="grok-beta", temperature=0))

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
    }
]
