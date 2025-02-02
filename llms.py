import os
import requests
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import keys  # keys.py or keysdummy.py with your credentials
from anthropic import Anthropic
from google import genai

# We define minimal message classes, so you don't need langchain_core's library
class HumanMessage:
    def __init__(self, content: str):
        self.content = content

class AIMessage:
    def __init__(self, content: str):
        self.content = content

class SystemMessage:
    def __init__(self, content: str):
        self.content = content

# ------------------------------------------------------------------------------
# HELPER: Token / length approximation for truncation
# ------------------------------------------------------------------------------
def approximate_tokens(text: str) -> int:
    """
    Very rough approximation of tokens, used for truncation only.
    We assume 1 token ~ 4 characters. You can refine with tiktoken if needed.
    """
    return max(1, len(text) // 4)

def approximate_total_tokens_for_messages(messages: List[Union[HumanMessage, AIMessage, SystemMessage]]) -> int:
    """
    Approximate total tokens for a list of messages.
    """
    total = 0
    for m in messages:
        total += approximate_tokens(m.content)
    return total


# ------------------------------------------------------------------------------
# BaseLLM – Our custom interface
# ------------------------------------------------------------------------------
class BaseLLM:
    """
    Base class for all custom LLM implementations.
    Must define an `invoke(messages, **kwargs)` method that returns
    an object with a `.content` attribute (the final text).
    """

    def __init__(self, temperature: float = 0.0, reasoning: bool = False, annotation: bool = False):
        self.temperature = max(0.0, min(2.0, temperature))
        self.reasoning = reasoning
        self.annotation = annotation

        # Because each LLM has different max context, we store that in the subclass.
        self.max_context_tokens = 4000  # By default, can be overridden

    def _truncate_messages(self, messages: List[Union[HumanMessage, AIMessage, SystemMessage]]) -> List[Union[HumanMessage, AIMessage, SystemMessage]]:
        """
        If the total tokens exceed the LLM's max context, drop from the earliest messages
        until they fit the max context limit. We also consider a small allowance for
        max_tokens in the completion (some LLMs need input+output <= limit).
        """
        # E.g., if a model's total limit is 128000, we might reserve 2000 for output
        # so that input + output doesn't exceed 128000. We do a simpler approach.
        reserve_for_output = int(self.max_context_tokens * 0.1)
        limit = self.max_context_tokens - reserve_for_output
        if limit < 1:
            limit = self.max_context_tokens  # fallback

        total = approximate_total_tokens_for_messages(messages)
        while total > limit and len(messages) > 1:
            # Drop the earliest message
            messages.pop(0)
            total = approximate_total_tokens_for_messages(messages)

        return messages

    def invoke(self, messages: List[Union[HumanMessage, AIMessage, SystemMessage]], **kwargs) -> AIMessage:
        """
        Must be implemented by each subclass. Returns AIMessage.
        """
        raise NotImplementedError("invoke() must be implemented in subclass.")


# ------------------------------------------------------------------------------
# DeepSeek Chat LLM Implementation
# ------------------------------------------------------------------------------
class DeepSeekChatLLM(BaseLLM):
    """
    Implementation for DeepSeek Chat using OpenAI SDK.
    Uses the standard chat model for general conversation.
    """
    def __init__(self, temperature: float = 0.0, reasoning: bool = False, annotation: bool = False):
        super().__init__(temperature=temperature, reasoning=reasoning, annotation=annotation)
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", keys.DEEPSEEK_API_KEY)
        # Initialize OpenAI client with DeepSeek base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        # DeepSeek has a 64K context length
        self.max_context_tokens = 64000

    def invoke(self, messages: List[Union[HumanMessage, AIMessage, SystemMessage]], **kwargs) -> AIMessage:
        # 1) Possibly truncate
        truncated = self._truncate_messages(messages)
        
        # 2) Convert to the format expected by DeepSeek
        payload_messages = []
        for m in truncated:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            payload_messages.append({"role": role, "content": m.content})

        # Default max_tokens is 4K, max is 8K for final response
        max_tokens = min(kwargs.get("max_tokens", 4096), 8192)

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=payload_messages,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            return AIMessage(content=content)
        except Exception as e:
            raise ValueError(f"DeepSeek Chat request failed: {e}")


# ------------------------------------------------------------------------------
# 1) DeepSeek R1 LLM Implementation
# ------------------------------------------------------------------------------
class DeepSeekR1LLM(BaseLLM):
    """
    Implementation for DeepSeek Reasoner using OpenAI SDK.
    Supports Chain of Thought (CoT) reasoning content on demand.
    """
    def __init__(self, temperature: float = 0.0, reasoning: bool = False, annotation: bool = False):
        super().__init__(temperature=temperature, reasoning=reasoning, annotation=annotation)
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", keys.DEEPSEEK_API_KEY)
        # Initialize OpenAI client with DeepSeek base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        # DeepSeek has a 64K context length
        self.max_context_tokens = 64000
        self.last_reasoning = None

    def invoke(self, messages: List[Union[HumanMessage, AIMessage, SystemMessage]], **kwargs) -> AIMessage:
        # 1) Possibly truncate
        truncated = self._truncate_messages(messages)
        
        # 2) Convert and interleave messages properly
        payload_messages = []
        system_content = ""
        
        # First, collect any system messages
        for m in truncated:
            if isinstance(m, SystemMessage):
                system_content += m.content + "\n"
        
        # If we have system content, add it as a system message at the start
        if system_content:
            payload_messages.append({"role": "system", "content": system_content.strip()})
        
        # Now handle the conversation messages, ensuring proper interleaving
        current_content = ""
        current_role = None
        
        for m in truncated:
            if isinstance(m, SystemMessage):
                continue  # Already handled above
                
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            
            # If this is the first message or if it's a different role than the current one
            if current_role is None:
                current_role = role
                current_content = m.content
            elif role != current_role:
                # Add the accumulated message and start a new one
                payload_messages.append({"role": current_role, "content": current_content})
                current_role = role
                current_content = m.content
            else:
                # Same role, combine the content
                current_content += "\n" + m.content
        
        # Add the last message if there is one
        if current_role is not None and current_content:
            payload_messages.append({"role": current_role, "content": current_content})
        
        # If the last message was from the assistant, add a dummy user message
        if payload_messages and payload_messages[-1]["role"] == "assistant":
            payload_messages.append({"role": "user", "content": "Please continue."})

        # If no messages after processing, add a default user message
        if not payload_messages:
            payload_messages.append({"role": "user", "content": "Hello"})

        # Default max_tokens is 4K, max is 8K for final response
        max_tokens = min(kwargs.get("max_tokens", 4096), 8192)

        try:
            # Print debug information
            print("\nDeepSeek Reasoner Request:")
            print(f"Number of messages: {len(payload_messages)}")
            print("Message roles sequence:", [m["role"] for m in payload_messages])
            
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=payload_messages,
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            
            # Print debug information about the response
            print("\nDeepSeek Reasoner Response:")
            print(f"Response status: Success")
            print(f"Number of choices: {len(response.choices)}")
            
            # Always store the reasoning content - it's part of the model's output
            try:
                self.last_reasoning = response.choices[0].message.reasoning_content
            except AttributeError:
                print("Warning: No reasoning_content found in response")
                self.last_reasoning = None
                
            content = response.choices[0].message.content
            return AIMessage(content=content)
        except Exception as e:
            print("\nDeepSeek Reasoner Error Details:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Last payload sent:")
            for i, msg in enumerate(payload_messages):
                print(f"Message {i + 1}:")
                print(f"  Role: {msg['role']}")
                print(f"  Content length: {len(msg['content'])}")
                print(f"  Content preview: {msg['content'][:100]}...")
            raise ValueError(f"DeepSeek Reasoner request failed: {e}")
    
    def get_last_reasoning(self) -> Optional[str]:
        """
        Returns the Chain of Thought (CoT) reasoning content from the last invoke call.
        """
        return self.last_reasoning


# ------------------------------------------------------------------------------
# 2) OpenAI o3-mini LLM
# ------------------------------------------------------------------------------
class OpenAIO3MiniLLM(BaseLLM):
    """
    Calls the OpenAI v1/chat/completions endpoint with model="o3-mini"
    """
    def __init__(self, temperature: float = 0.0, reasoning: bool = False, annotation: bool = False):
        super().__init__(temperature=temperature, reasoning=reasoning, annotation=annotation)
        self.api_key = os.environ.get("OPENAI_API_KEY", keys.OPENAI_KEY)
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        # o3-mini supports 200,000 tokens total (input + output)
        self.max_context_tokens = 200000

    def invoke(self, messages: List[Union[HumanMessage, AIMessage, SystemMessage]], **kwargs) -> AIMessage:
        truncated = self._truncate_messages(messages)

        # Convert messages
        payload_messages = []
        for m in truncated:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            payload_messages.append({"role": role, "content": m.content})

        # Default max_completion_tokens and top_p
        max_completion_tokens = kwargs.get("max_tokens", 90000)  # Default to 90k, max is 100k
        top_p = kwargs.get("top_p", 1.0)

        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=payload_messages,
                max_completion_tokens=max_completion_tokens,
                top_p=top_p
            )
            content = response.choices[0].message.content
            return AIMessage(content=content)
        except Exception as e:
            raise ValueError(f"OpenAI o3-mini request failed: {e}")


# ------------------------------------------------------------------------------
# 3) OpenAI GPT-4o LLM
# ------------------------------------------------------------------------------
class OpenAIGPT4oLLM(BaseLLM):
    """
    Calls the OpenAI v1/chat/completions endpoint with model="gpt-4o"
    """
    def __init__(self, temperature: float = 0.0, reasoning: bool = False, annotation: bool = False):
        super().__init__(temperature=temperature, reasoning=reasoning, annotation=annotation)
        self.api_key = os.environ.get("OPENAI_API_KEY", keys.OPENAI_KEY)
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        # GPT-4o typically allows up to ~128000 input tokens, plus up to 16384 output tokens
        # We'll set 128000 as an overall input limit to be safe
        self.max_context_tokens = 128000
        self.max_completion_tokens = 16384  # Maximum allowed completion tokens for GPT-4

    def invoke(self, messages: List[Union[HumanMessage, AIMessage, SystemMessage]], **kwargs) -> AIMessage:
        truncated = self._truncate_messages(messages)

        payload_messages = []
        for m in truncated:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            payload_messages.append({"role": role, "content": m.content})

        # Ensure max_tokens doesn't exceed the model's limit
        max_tokens = min(kwargs.get("max_tokens", self.max_completion_tokens), self.max_completion_tokens)
        top_p = kwargs.get("top_p", 1.0)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=payload_messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            content = response.choices[0].message.content
            return AIMessage(content=content)
        except Exception as e:
            raise ValueError(f"OpenAI GPT-4o request failed: {e}")


# ------------------------------------------------------------------------------
# 4) Anthropic Claude LLM
# ------------------------------------------------------------------------------
class AnthropicClaudeLLM(BaseLLM):
    """
    Calls the Anthropic v1/messages endpoint with model="claude-3-5-sonnet-latest"
    """
    def __init__(self, temperature: float = 0.0, reasoning: bool = False, annotation: bool = False):
        super().__init__(temperature=temperature, reasoning=reasoning, annotation=annotation)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", keys.CLAUDE_KEY)
        self.client = Anthropic(api_key=self.api_key)
        # Claude has a 200k token limit (input + output)
        self.max_context_tokens = 200000
        self.max_completion_tokens = 8192  # Maximum allowed completion tokens for Claude 3.5 Sonnet

    def invoke(self, messages: List[Union[HumanMessage, AIMessage, SystemMessage]], **kwargs) -> AIMessage:
        truncated = self._truncate_messages(messages)

        # Convert messages to Anthropic format
        payload_messages = []
        system_message = ""
        for m in truncated:
            if isinstance(m, SystemMessage):
                system_message += m.content + "\n"
            else:
                role = "user" if isinstance(m, HumanMessage) else "assistant"
                payload_messages.append({"role": role, "content": m.content})

        # Ensure max_tokens doesn't exceed the model's limit
        max_tokens = min(kwargs.get("max_tokens", self.max_completion_tokens), self.max_completion_tokens)

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-latest",
                messages=payload_messages,
                system=system_message if system_message else None,
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            content = response.content[0].text
            return AIMessage(content=content)
        except Exception as e:
            raise ValueError(f"Anthropic Claude request failed: {e}")


# ------------------------------------------------------------------------------
# Placeholders for other model types that your code references
# ------------------------------------------------------------------------------

# --- Gemini LLM Implementation ---
class GeminiLLM(BaseLLM):
    """
    Implementation for Google's Gemini Advanced model using the official Google Generative AI client.
    Uses model "gemini-1.5-pro" for general use.
    """
    def __init__(self, temperature: float = 0.0, reasoning: bool = False, annotation: bool = False):
        super().__init__(temperature=max(0.0, min(1.0, temperature)), reasoning=reasoning, annotation=annotation)
        # Use GEMINI_API_KEY from keys.py
        self.api_key = os.environ.get("GEMINI_API_KEY", keys.GEMINI_API_KEY)
        # Initialize the Gemini client
        self.client = genai.Client(api_key=self.api_key)
        self.max_context_tokens = 1_000_000  # Gemini has a large context window
        self.model_name = "gemini-1.5-pro"

    def invoke(self, messages: List[Union[HumanMessage, AIMessage, SystemMessage]], **kwargs) -> AIMessage:
        truncated = self._truncate_messages(messages)

        # Combine messages into a single conversation string
        # For Gemini, we'll format system messages as special instructions
        combined_content = ""
        for m in truncated:
            if isinstance(m, SystemMessage):
                combined_content += f"[System Instructions: {m.content}]\n"
            elif isinstance(m, HumanMessage):
                combined_content += f"User: {m.content}\n"
            else:  # AIMessage
                combined_content += f"Assistant: {m.content}\n"

        try:
            # Generate content using the model
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=combined_content
            )
            
            # Extract the response text
            content = response.candidates[0].content.parts[0].text
            return AIMessage(content=str(content))
        except Exception as e:
            raise ValueError(f"Google Gemini request failed: {e}")

# --- Gemini 2.0 Flash Thinking LLM Implementation ---
class GeminiFlashThinkingLLM(BaseLLM):
    """
    Implementation for Gemini 2.0 Flash Thinking Experimental API using the official Google Generative AI client.
    Uses model "gemini-2.0-flash-thinking-exp" with support for multi-turn conversations and thinking process.
    """
    def __init__(self, temperature: float = 0.0, reasoning: bool = False, annotation: bool = False):
        super().__init__(temperature=max(0.0, min(1.0, temperature)), reasoning=reasoning, annotation=annotation)
        # Use GEMINI_API_KEY from keys.py
        self.api_key = os.environ.get("GEMINI_API_KEY", keys.GEMINI_API_KEY)
        # Initialize the Gemini client
        self.client = genai.Client(
            api_key=self.api_key,
            http_options={'api_version': 'v1alpha'}
        )
        self.max_context_tokens = 1_000_000
        self.model_name = "gemini-2.0-flash-thinking-exp"
        self.last_thoughts = None

    def invoke(self, messages: List[Union[HumanMessage, AIMessage, SystemMessage]], **kwargs) -> AIMessage:
        truncated = self._truncate_messages(messages)

        # Combine messages into a single conversation string
        # For Gemini, we'll format system messages as special instructions
        combined_content = ""
        for m in truncated:
            if isinstance(m, SystemMessage):
                combined_content += f"[System Instructions: {m.content}]\n"
            elif isinstance(m, HumanMessage):
                combined_content += f"User: {m.content}\n"
            else:  # AIMessage
                combined_content += f"Assistant: {m.content}\n"

        try:
            # Generate content using the model with thinking config
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=combined_content,
                config=genai.types.GenerateContentConfig(
                    temperature=self.temperature,
                    candidate_count=1,
                    thinking_config=genai.types.ThinkingConfig(
                        include_thoughts=True
                    )
                )
            )
            
            # Extract thoughts and final content
            self.last_thoughts = []
            final_content = ""
            
            # Process each part of the response
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'is_thought') and part.is_thought:
                    self.last_thoughts.append(part.text)
                else:
                    final_content += part.text
            
            return AIMessage(content=final_content)
        except Exception as e:
            raise ValueError(f"Google Gemini request failed: {e}")
    
    def get_last_thoughts(self) -> Optional[List[str]]:
        """
        Returns the thinking process from the last invoke call.
        """
        return self.last_thoughts

# --- xAI Grok LLM Implementation ---
class GrokLLM(BaseLLM):
    """
    Implementation for xAI Grok API.
    Calls:
      POST https://api.x.ai/v1/chat/completions
    with model "grok-beta".
    
    Supports multi-turn conversations similar to OpenAI chat completions.
    """
    def __init__(self, temperature: float = 0.0, reasoning: bool = False, annotation: bool = False):
        super().__init__(temperature=temperature, reasoning=reasoning, annotation=annotation)
        # Grok API key: ensure it's set in your environment or keys module.
        self.api_key = os.environ.get("XAI_API_KEY", getattr(keys, "GROK_API_KEY", None))
        self.max_context_tokens = 128000
        self.model_name = "grok-beta"

    def invoke(self, messages: List[Union[HumanMessage, AIMessage, SystemMessage]], **kwargs) -> AIMessage:
        truncated = self._truncate_messages(messages)
        payload_messages = []
        for m in truncated:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            payload_messages.append({"role": role, "content": m.content})
        max_tokens = kwargs.get("max_tokens", 1024)
        top_p = kwargs.get("top_p", 1.0)
        body = {
            "model": self.model_name,
            "messages": payload_messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        url = "https://api.x.ai/v1/chat/completions"
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=300)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(f"xAI Grok request failed: {e}")
        data = resp.json()
        if "choices" not in data or not data["choices"]:
            raise ValueError(f"xAI Grok: No choices returned. Response: {data}")
        content = data["choices"][0]["message"]["content"]
        return AIMessage(content=content)


class OpenAIGPTo1LLM(BaseLLM):
    """
    Placeholder for "openai_gpt-o1" (o1-preview).
    The original code mentions 'o1-preview doesn't support system messages and only supports temperature=1'.
    We'll skip direct API logic and raise an error.
    """
    def invoke(self, messages: List[Union[HumanMessage, AIMessage, SystemMessage]], **kwargs) -> AIMessage:
        raise NotImplementedError("OpenAIGPTo1LLM (o1-preview) is not yet implemented.")


# ------------------------------------------------------------------------------
# SystemMessageHandlingLLM
# ------------------------------------------------------------------------------
# --- Existing Wrappers and Factory (updated) ---
class SystemMessageHandlingLLM:
    def __init__(self, llm, supports_system_messages: bool = True):
        self.llm = llm
        self.supports_system_messages = supports_system_messages

    def _convert_messages(self, messages):
        if self.supports_system_messages:
            return messages
        converted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted.append(HumanMessage(content=f"System instruction: {msg.content}"))
            else:
                converted.append(msg)
        return converted

    def invoke(self, messages, **kwargs):
        if not messages:
            return AIMessage(content="")
        messages = self._convert_messages(messages)
        return self.llm.invoke(messages, **kwargs)

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
    def invoke(self, messages, **kwargs):
        try:
            return self.llm.invoke(messages, **kwargs)
        except Exception as e:
            s = str(e).lower()
            if "429" in s or "rate limit" in s:
                print("Rate limit hit. Retrying with exponential backoff...")
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        reraise=True
    )
    def batch(self, *args, **kwargs):
        raise NotImplementedError("Batch calls not implemented in custom LLM classes yet.")

    def __getattr__(self, name):
        return getattr(self.llm, name)

def create_llm(model_type: str, temperature: float = 0) -> RetryLLM:
    temperature = max(0.0, min(2.0, temperature))
    supports_system = True
    if model_type == "deepseek_chat":
        raw_llm = DeepSeekChatLLM(temperature=temperature)
        wrapped = SystemMessageHandlingLLM(raw_llm, supports_system_messages=True)
        return RetryLLM(wrapped)
    elif model_type == "deepseek_r1":
        raw_llm = DeepSeekR1LLM(temperature=temperature)
        wrapped = SystemMessageHandlingLLM(raw_llm, supports_system_messages=True)
        return RetryLLM(wrapped)
    elif model_type == "openai_gpt4o":
        raw_llm = OpenAIGPT4oLLM(temperature=temperature)
        wrapped = SystemMessageHandlingLLM(raw_llm, supports_system_messages=True)
        return RetryLLM(wrapped)
    elif model_type == "anthropic_claude":
        raw_llm = AnthropicClaudeLLM(temperature=temperature)
        wrapped = SystemMessageHandlingLLM(raw_llm, supports_system_messages=True)
        return RetryLLM(wrapped)
    elif model_type == "openai_gpt-o1":
        raw_llm = OpenAIGPTo1LLM(temperature=1.0)
        wrapped = SystemMessageHandlingLLM(raw_llm, supports_system_messages=False)
        return RetryLLM(wrapped)
    elif model_type == "google_gemini":
        raw_llm = GeminiLLM(temperature=temperature)
        wrapped = SystemMessageHandlingLLM(raw_llm, supports_system_messages=True)
        return RetryLLM(wrapped)
    elif model_type == "gemini_flash_thinking":
        raw_llm = GeminiFlashThinkingLLM(temperature=temperature)
        wrapped = SystemMessageHandlingLLM(raw_llm, supports_system_messages=True)
        return RetryLLM(wrapped)
    elif model_type == "xai_grok":
        raw_llm = GrokLLM(temperature=temperature)
        wrapped = SystemMessageHandlingLLM(raw_llm, supports_system_messages=True)
        return RetryLLM(wrapped)
    elif model_type in ["openai_gpt-o3-mini", "openai_o3_mini"]:
        raw_llm = OpenAIO3MiniLLM(temperature=temperature)
        wrapped = SystemMessageHandlingLLM(raw_llm, supports_system_messages=True)
        return RetryLLM(wrapped)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# --- Pre-configured LLM Instances (updated) ---
openai_llm = create_llm("openai_gpt4o", temperature=0.0)
openai_llm_reasoning = create_llm("openai_gpt-o3-mini", temperature=1.0)
deepseek_chat_llm = create_llm("deepseek_chat", temperature=0.0)
deepseek_r1_llm = create_llm("deepseek_r1", temperature=0.0)
claude_llm = create_llm("anthropic_claude", temperature=0.0)
gemini_llm = create_llm("google_gemini", temperature=0.0)
grok_llm = create_llm("xai_grok", temperature=0.0)
gemini_llm_flash = create_llm("gemini_flash_thinking", temperature=0.0)

cheap_llm = deepseek_chat_llm
sota_llm = openai_llm

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
        "description": "Google's Gemini 1.5 Pro model for general use.",
        "llm": gemini_llm
    },
    {
        "name": "gemini_flash_thinking",
        "description": "Google's Gemini 2.0 Flash Thinking model with reasoning support.",
        "llm": gemini_llm_flash
    },
    {
        "name": "xai_grok",
        "description": "xAI's Grok model, an AI chatbot integrated with X, suitable for real-time and multimodal tasks.",
        "llm": grok_llm
    },
    {
        "name": "deepseek_chat",
        "description": "DeepSeek's standard chat model for general conversation.",
        "llm": deepseek_chat_llm
    },
    {
        "name": "deepseek_r1",
        "description": "DeepSeek's R1 model with Chain of Thought reasoning capabilities.",
        "llm": deepseek_r1_llm
    },
    {
        "name": "openai_gpt-o3-mini",
        "description": "Open AI's o3-mini reasoning model).",
        "llm": openai_llm_reasoning
    }
]

if __name__ == "__main__":
    # Define a test conversation input (a simple math reasoning prompt)
    test_input_math = [
        SystemMessage("You are an expert mathematician."),
        HumanMessage("Solve the equation 3x^3 - 5x = 1 and explain your reasoning.")
    ]
    
    # Define a test conversation input for coding
    test_input_code = [
        SystemMessage("You are an expert coding assistant."),
        HumanMessage("Implement Depth First Search in Python.")
    ]
    
    # Loop through all available LLMs in available_llms and test them.
    for llm_info in available_llms:
        name = llm_info.get("name")
        description = llm_info.get("description")
        llm_instance = llm_info.get("llm")
        
        print("\n========================================")
        print(f"Testing LLM: {name}")
        print(f"Description: {description}")
        
        try:
            # For demonstration, use the math test input for reasoning models
            # and the coding test input for models that might be better suited for coding.
            if "deepseek_r1" in name:
                # Enable reasoning for DeepSeek to get Chain of Thought
     
                test_input = test_input_math
                max_tokens = 4096
                
                # Get response
                response = llm_instance.invoke(test_input, max_tokens=max_tokens)
                
                # Print both Chain of Thought and final answer
                print(f"\nChain of Thought from {name}:")
                print(llm_instance.get_last_reasoning())
                print(f"\nFinal Answer from {name}:")
                print(response.content)
            elif "gemini_flash_thinking" in name:
                continue
                test_input = test_input_math
                max_tokens = 4096
                
                # Get response
                response = llm_instance.invoke(test_input, max_tokens=max_tokens)
                
                # Print both thoughts and final answer
                print(f"\nThinking Process from {name}:")
                thoughts = llm_instance.get_last_thoughts()
                if thoughts:
                    for i, thought in enumerate(thoughts, 1):
                        print(f"Thought {i}: {thought}")
                print(f"\nFinal Answer from {name}:")
                print(response.content)
            else:
                if "deepseek" not in name:
                    continue
                # Handle other models normally
                if "gpt4o" in name or "gpt-o3-mini" in name or "claude" in name:
                    test_input = test_input_math

                    max_tokens = 4096
                elif "grok" in name:
                    test_input = test_input_code
                    max_tokens = 1024
                else:
                    test_input = test_input_math
                    max_tokens = 2048

                response = llm_instance.invoke(test_input, max_tokens=max_tokens)
                print(f"Response from {name}:\n{response.content}")
        except Exception as e:
            print(f"Error testing {name}: {e}")