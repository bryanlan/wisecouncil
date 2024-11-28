from typing import List, Union, TypedDict, Dict, Any, Tuple
from langgraph.graph import Graph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
from langchain_community.tools.tavily_search import TavilySearchResults
import trafilatura
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time
import base64
import random
import gradio as gr
import os
import io
import keys
import json
import re
import copy
import datetime
from readability import Document
import requests
import justext
import html2text
from PIL import Image
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import quote_plus
from tenacity import retry, stop_after_attempt, wait_exponential
import traceback

# Debugging flag
debugme = 1  # Set to 0 to disable debugging window
moderatorPromptEnd = (f"Respond in the following JSON format:\n"
    "{\n"
    '   "next_agent": "the name of the next agent to speak",\n'
    '   "agent_instruction": "Name of the next agent followed by instructions to the agent",\n'
    '   "final_thoughts": "Your final thoughts on the conversation.",\n'
    "}"
           
        )

TOOLPREFIX = "Tool Provided Data:"

def clean_response(content: str) -> str:
    """
    Cleans the response content by:
    - Removing code fences and JSON labels
    - Removing outer quotes if present
    - Fixing invalid escape sequences
    - Ensuring the string can be parsed as valid JSON
    """
    # Remove code fences and JSON labels
    content = content.strip()
    if content.startswith('```'):
        content = content.strip('`')
        content = re.sub(r'^json\n', '', content, flags=re.IGNORECASE).strip()

    # Remove outer quotes if present
    if (content.startswith("'") and content.endswith("'")) or \
       (content.startswith('"') and content.endswith('"')):
        content = content[1:-1]

    # Replace invalid escape sequences
    # Replace \' with '
    content = content.replace("\\'", "'")

    # Remove any remaining backslashes not part of valid escape sequences
    # Valid escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    # We'll use a regex to find invalid ones and remove the backslash
    def fix_invalid_escapes(match):
        escape = match.group(0)
        if re.match(r'\\u[0-9a-fA-F]{4}', escape):
            return escape  # Valid Unicode escape sequence
        elif escape in ('\\"', '\\\\', '\\/', '\\b', '\\f', '\\n', '\\r', '\\t'):
            return escape  # Valid escape sequence
        else:
            return escape[1:]  # Remove the backslash

    content = re.sub(r'\\.', fix_invalid_escapes, content)

    # Now, try to load the JSON to check if it's valid
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decoding failed: {e.msg} at line {e.lineno} column {e.colno} (char {e.pos})")

    return content

# Ensure you have set these environment variables
os.environ["OPENAI_API_KEY"] = keys.OPENAI_KEY
os.environ["ANTHROPIC_API_KEY"] = keys.CLAUDE_KEY
os.environ["TAVILY_API_KEY"] = keys.TAVILY_KEY
os.environ["GOOGLE_API_KEY"] = keys.GOOGLE_API_KEY
os.environ["XAI_API_KEY"] = keys.GROK_API_KEY
os.environ["GOOGLE_SEARCH_ID"] = keys.GOOGLE_SEARCH_ENGINE_ID
GEMINI_KEY = keys.GEMINI_API_KEY

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
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self.llm.invoke(*args, **kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    def batch(self, *args: Any, **kwargs: Any) -> Any:
        return self.llm.batch(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.llm, name)

# Initialize your LLMs with retry wrapper
openai_llm = RetryLLM(ChatOpenAI(model="gpt-4o", temperature=0))
openai_llm_mini = RetryLLM(ChatOpenAI(model="gpt-4o-mini", temperature=0))
claude_llm = RetryLLM(ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0))
gemini_llm = RetryLLM(ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0,api_key=GEMINI_KEY))
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
    # {
    #    "name": "google_gemini",
    #    "description": "Google's Gemini model, a state-of-the-art multimodal AI model.",
    #    "llm": gemini_llm
    #},
     {
        "name": "xai_grok",
        "description": "xAI's Grok model, an AI chatbot integrated with X. Best for espousing controversial viewpoints.",
        "llm": grok_llm
    }

]

class Tool:
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func

    def invoke(self, input_data: str) -> str:
        return self.func(input_data)
    
class ResearchTool(Tool):
    def __init__(self,max_total_image_interpretations: int = 10, max_interpretations_per_page: int = 4, dismissPopups=True):
        super().__init__(
            name="do_research",
            description="Perform internet research on a given query",
            func=self._research_func
        )
        self.api_key = keys.GOOGLE_API_KEY
        self.search_engine_id = keys.GOOGLE_SEARCH_ENGINE_ID
        self.setup_driver()
        self.service = build("customsearch", "v1", developerKey=self.api_key)
        self.dismissPopups = dismissPopups
        
        # Common problematic domains that might block scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Inside the __init__ method of ResearchTool
        # Inside the __init__ method of ResearchTool
        self.popup_selectors = [
        "button.accept-cookies",
        "button.cookie-consent-accept",
        "button#accept",
        "button#cookieAccept",
        "button[class*='accept']",
        "button[class*='agree']",
        "button[class*='consent']",
        "button[class*='close']",
        ".cookie-consent button",
        ".cookie-banner button",
        ".modal-footer button.btn-primary",
        ".popup-close",
        ".overlay-close",
        # Add more selectors as needed
    ]

        self.alert_texts = [
            "Allow",
            "Deny",
            "Block",
            "No Thanks",
            "Not Now",
            "Cancel",
            "Close",
            "Reject",
            "Decline",
            "Got it",
            "OK",
            # Add more texts as needed
        ]

        # Initialize interpretation limits and counters
        self.max_total_image_interpretations = max_total_image_interpretations
        self.max_interpretations_per_page = max_interpretations_per_page
        self.total_image_interpretations = 0
        # Initialize extracted texts accumulator
        self.accumulated_extracted_texts = []  # Accumulates all extracted texts per search


    

    def _dismiss_popups(self):
        """
        Attempts to dismiss common popups/dialogs on the current page.
        """
        max_attempts = 3  # Number of times to attempt popup dismissal
        wait_time = 5     # Increased wait time to 5 seconds
        attempts = 0

        try:
            while attempts < max_attempts:
                popup_found = False

                # Handle JavaScript alerts
                try:
                    WebDriverWait(self.driver, wait_time).until(EC.alert_is_present())
                    alert = self.driver.switch_to.alert
                    alert.dismiss()
                    print("Dismissed JavaScript alert.")
                    popup_found = True
                except:
                    pass

                # Dismiss popups via CSS selectors
                for selector in self.popup_selectors:
                    try:
                        elements = WebDriverWait(self.driver, wait_time).until(
                            EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                        )
                        for element in elements:
                            if element.is_displayed() and element.is_enabled():
                                element.click()
                                print(f"Clicked popup button with selector: {selector}")
                                time.sleep(0.5)
                                popup_found = True
                    except:
                        continue  # Continue to next selector if none found

                # Dismiss popups via button texts using XPath
                for text in self.alert_texts:
                    try:
                        xpath = f"//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]"
                        elements = WebDriverWait(self.driver, wait_time).until(
                            EC.presence_of_all_elements_located((By.XPATH, xpath))
                        )
                        for element in elements:
                            if element.is_displayed() and element.is_enabled():
                                element.click()
                                print(f"Clicked popup button with text: '{text}'")
                                time.sleep(0.5)
                                popup_found = True
                    except:
                        continue  # Continue to next text if none found

                # Optional: Remove popups via JavaScript
                self._remove_popups_via_js()

                # Break the loop if no popups were found in this iteration
                if not popup_found:
                    break

                attempts += 1

        except Exception as e:
            print(f"Error while attempting to dismiss popups: {str(e)}")

    def _remove_popups_via_js(self):
        try:
            scripts = [
                "document.querySelectorAll('.overlay, .modal, .popup, #overlay, #modal, #popup').forEach(el => el.remove());",
                "document.querySelectorAll('div[class*=\"cookie\"], div[class*=\"consent\"], div[class*=\"banner\"]').forEach(el => el.style.display='none');",
                "document.querySelectorAll('button[class*=\"close\"], button[class*=\"dismiss\"]').forEach(el => el.click());",
            ]
            for script in scripts:
                self.driver.execute_script(script)
                print("Executed JS to remove popups.")
                time.sleep(0.5)
        except Exception as e:
            print(f"Error executing JS to remove popups: {str(e)}")


    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36')
        
        # Add these options to reduce noise
        chrome_options.add_argument('--log-level=3')  # Fatal only
        chrome_options.add_argument('--silent')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

        # Add the unsafe swiftshader flag
        chrome_options.add_argument('--enable-unsafe-swiftshader')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

    def _extract_content(self, url: str, title: str, snippet: str, query: str) -> tuple:
        try:
            image_reading_needed = False
            self.driver.get(url)
            print(f"Navigated to URL: {url}")
            time.sleep(2)  # Wait for the page to load

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Use readability to get the main content
            doc = Document(response.text)
            
            # Try to extract publication date from meta tags first
            pub_date = None
            confidence = 'low'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check common meta tags for date
            for meta in soup.find_all('meta'):
                property_val = meta.get('property', '')
                name_val = meta.get('name', '')
                if any(date_tag in (property_val, name_val) for date_tag in [
                    'article:published_time',
                    'datePublished',
                    'publication_date',
                    'date',
                    'pubdate'
                ]):
                    pub_date = meta.get('content')
                    confidence = 'high'
                    break
            
            # Get the main content
            content = doc.summary()
            
            # Use justext to remove boilerplate content
            paragraphs = justext.justext(content, justext.get_stoplist("English"))
            content_parts = []
            for paragraph in paragraphs:
                if not paragraph.is_boilerplate:
                    content_parts.append(paragraph.text)
            
            # Convert HTML to plain text
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            h.ignore_emphasis = True
            text_content = h.handle(' '.join(content_parts))
            
            # Clean up the text
            clean_text = self._clean_text(text_content)

            # If no text content was extracted, set image_reading_needed flag
            if not clean_text.strip():
                print(f"No text content extracted from {url}, will defer image reading.")
                image_reading_needed = True
                # Do not proceed to _extract_content_with_scrolling()
              
            # If no date found in meta tags, use GPT to extract date using the full content
            if not pub_date and not image_reading_needed:
                current_date = datetime.datetime.now()
                pub_date, confidence = self._extract_date_with_gpt(
                    url=url,
                    title=title,
                    snippet=snippet,
                    content=clean_text,
                    current_date=current_date
                )
            
            if not image_reading_needed:
                 # Format and prepend title and date if available
                formatted_text = f"Title: {title}\n"
                if pub_date:
                    formatted_text += f"Date: {pub_date}\n"
                formatted_text += f"Content:\n{clean_text}\n"
            
            # Accumulate the formatted text
                self.accumulated_extracted_texts.append(formatted_text)

            
            return clean_text, pub_date, confidence, image_reading_needed

        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
            return "", None, 'low', False

    def _check_continue_interpretation(self, total_info: str, current_page_info: str, query: str) -> bool:
        try:
            prompt = (
                f"Here is the information extracted so far from all pages:\n{total_info}\n\n"
                f"Here is the information extracted from the current page:\n{current_page_info}\n\n"
                f"The original query is: '{query}'.\n\n"
                f"Based on the information extracted, should we perform another image interpretation to extract more information? Say yes if you believe more useful information to help the query is likely to come from the current page."
                f"Respond with ONLY 'yes - we should do another extraction' or 'no - we've received all of the useful info' and a brief description of your reason."
            )
            
            response = openai_llm_mini.invoke([HumanMessage(content=prompt)])
            decision = response.content.strip().lower()
            
            if 'yes' in decision:
                print(f"Will do another extraction because {decision}")
                return True
                
            elif 'no' in decision:
                print(f"Will not do another extraction because {decision}")
                return False
            else:
                # Default to not continue if unclear
                print("GPT response unclear. Defaulting to stop further interpretations.")
                return False

        except Exception as e:
            print(f"Error during GPT decision-making: {str(e)}")
            # Default to not continue on error
            return False

    def _extract_content_with_scrolling(self, url: str, query: str) -> str:
        try:
            self.driver.get(url)
            time.sleep(2)  # Wait for the page to load

            extracted_texts = []
            current_page_image_interpretations = 0
            max_screens = self.max_interpretations_per_page

            for screen in range(max_screens):
                if self.total_image_interpretations >= self.max_total_image_interpretations:
                    print("Reached maximum total image interpretations.")
                    break

                # Capture screenshot
                screenshot = self.driver.get_screenshot_as_png()
                print(f"Captured screenshot {screen + 1} for {url}")
                
                # Extract text from the screenshot
                text = self._extract_text_from_image_with_gpt(screenshot, query=query)
                if text:
                    extracted_texts.append(text)
                    self.total_image_interpretations += 1
                    current_page_image_interpretations += 1
                    

                    
                    # Check with GPT whether to continue
                    total_info = "\n".join(self.accumulated_extracted_texts)  # Accumulated across all pages
                    current_page_info = "\n".join(extracted_texts)  # Accumulated on current page
                    
                    should_continue = self._check_continue_interpretation(
                        total_info=total_info,
                        current_page_info=current_page_info,
                        query=query
                    )
                    
                    if not should_continue:
                        print("GPT advised to stop further image interpretations.")
                        break

                # Scroll down by the viewport height
                if screen < max_screens - 1:
                    self._scroll_down()
                    time.sleep(2)  # Wait for new content to load

            # Combine all extracted texts
            combined_text = "\n".join(extracted_texts)
            return combined_text

        except Exception as e:
            print(f"Error extracting content with scrolling from {url}: {str(e)}")
            return ""

    def _scroll_down(self):
        try:
            # Scroll down by the viewport height
            scroll_script = "window.scrollBy(0, window.innerHeight);"
            self.driver.execute_script(scroll_script)
            print("Scrolled down by one viewport height.")
        except Exception as e:
            print(f"Error scrolling down: {str(e)}")

    def _extract_text_from_image_with_gpt(self, image_data: bytes, query: str) -> str:
        
        bmp_filename = f"screenshot_{int(time.time())}.bmp"
        try:
            # Encode the image data to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            # Save the image data as a BMP file for debugging
            image = Image.open(io.BytesIO(image_data))
            image.save(bmp_filename, format='BMP')
            print(f"Saved BMP file for debugging: {bmp_filename}")

            # Create the message with the image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": f"Please extract all textual information from this image that is relevant to query:'{query}' and provide it as plain text. Respond with ONLY the extracted text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ]
            )

            # Invoke the model with the message
            response = openai_llm_mini.invoke([message])

            # Extract and return the text from the response
            extracted_text = response.content.strip()
            print("Extracted text from image.")
            return extracted_text

        except Exception as e:
            print(f"Error extracting text from image with GPT: {str(e)}")
            return ""
        finally:
        # Delete the BMP file immediately after processing
            try:
                if os.path.exists(bmp_filename):
                    os.remove(bmp_filename)
                    print(f"Deleted BMP file: {bmp_filename}")
            except Exception as del_e:
                print(f"Error deleting BMP file: {str(del_e)}")
        
    def _clean_text(self, text: str) -> str:
        # Split into lines and clean
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines or very short lines
            if len(line) < 4:
                continue
            # Skip navigation-like items
            if len(line.split()) < 3:
                continue
            # Skip lines that are likely navigation or UI elements
            if any(nav in line.lower() for nav in ['menu', 'search', 'skip to', 'cookie', 'privacy policy']):
                continue
            cleaned_lines.append(line)
        
        # Join lines back together
        text = ' '.join(cleaned_lines)
        
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        return text
    def _google_search(self, query: str, required_terms: list, excluded_terms: list, time_range: str, num_results: int = 10):
        print(f"Starting google_search with query: {query}") 
        try:
            # Build search query
            search_query = query + " -site:youtube.com -site:youtu.be -site:reddit.com -site:washingtonpost.com"

            url = f"https://cse.google.com/cse?cx={self.search_engine_id}&q={quote_plus(search_query)}"
            print(f"Searching URL: {url}")
            
            self.driver.get(url)
            
            # Wait for results
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "gsc-result"))
            )
            
            time.sleep(3)

            # Use JavaScript to extract the results
            js_script = """
            const results = [];
            const elements = document.querySelectorAll('.gsc-result');
            elements.forEach(element => {
                const titleElement = element.querySelector('a.gs-title');
                const snippetElement = element.querySelector('.gs-snippet');
                if (titleElement && titleElement.href) {
                    results.push({
                        url: titleElement.href,
                        title: titleElement.textContent,
                        snippet: snippetElement ? snippetElement.textContent : ''
                    });
                }
            });
            return results;
            """
            
            search_results = self.driver.execute_script(js_script)
            print(f"Found {len(search_results)} results")
            if (len(search_results) >0):
                
                results = []
                deferred_results = []
                processed_urls = set()
                
                for result in search_results:
                    try:
                        url = result['url']
                        if not url or not url.startswith('http'):
                            continue
                            
                        if url in processed_urls:
                            continue
                            
                        processed_urls.add(url)
                        
                        if any(domain in url.lower() for domain in [
                            'youtube.com', 
                            'youtu.be',
                            'vimeo.com',
                            'dailymotion.com',
                            'tiktok.com'
                        ]):
                            continue
                        
                        title = result['title']
                        snippet = result['snippet']
                        
                        print(f"Processing URL: {url}")                    
                        full_content, pub_date, confidence, image_reading_needed = self._extract_content(url, title, snippet, query=query)

                        if image_reading_needed:
                            # Defer processing this result
                            deferred_results.append(result)
                            continue

                        if full_content and len(full_content.split()) > 50:
                            print(f"Adding result for: {url}")
                            results.append({
                                'url': url,
                                'title': title,
                                'snippet': snippet,
                                'content': full_content,
                                'date': pub_date,
                                'date_confidence': confidence
                            })
                            print(f"Current results count: {len(results)}")
                            
                    except Exception as e:
                        print(f"Error processing result: {e}")
                        continue

                print(f"Processed {len(results)} results without image reading")
                print(f"Deferred {len(deferred_results)} results that require image reading")

                # Decide whether to process deferred results
                should_process_deferred = self._decide_to_process_deferred(query, time_range)

                if should_process_deferred:
                    print("Proceeding to process deferred results with image reading.")
                    for result in deferred_results:
                        self._process_deferred_result(result, query)
                else:
                    print("Skipping processing of deferred results as sufficient information is available.")

                print(f"Returning {len(results)} results")
                return results
            else:
                print("Google search needs to have not a robot clicked, exiting")
                exit()

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def _decide_to_process_deferred(self, query: str, time_range: str) -> bool:
        accumulated_info = "\n".join(self.accumulated_extracted_texts)
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        prompt = (
            f"Given the following query: '{query}', today's date: {current_date}, time sensitivity of the required data: '{time_range}',\n"
            f"and the information extracted so far:\n"
            f"{accumulated_info}\n\n"
            f"Do you have enough information to answer the query confidently?\n"
            f"Would you benefit from taking significant extra processing time to extract more information from additional sources?\n"
            f"Respond with 'yes - more info is needed' or 'no - I have enough info' and a brief justification."
        )
        response = openai_llm_mini.invoke([HumanMessage(content=prompt)])
        decision = response.content.strip().lower()
        if 'yes' in decision:
            return True
        elif 'no' in decision:
            return False
        else:
            # Default to False if the response is unclear
            print("GPT response unclear. Defaulting to not process deferred results.")
            return False

    def _process_deferred_result(self, result, query):
        url = result['url']
        title = result['title']
        snippet = result['snippet']

        print(f"Processing deferred URL with image reading: {url}")
        full_content = self._extract_content_with_scrolling(url, query=query)
        if not full_content.strip():
            print(f"Could not extract content from {url} even with image reading.")
            return

        # Accumulate the cleaned text
        self.accumulated_extracted_texts.append(full_content)

        # Try to extract date
        pub_date = None
        confidence = 'low'
        current_date = datetime.datetime.now()
        pub_date, confidence = self._extract_date_with_gpt(
            url=url,
            title=title,
            snippet=snippet,
            content=full_content,
            current_date=current_date
        )

        # Add the result to the main results list
        self.results.append({
            'url': url,
            'title': title,
            'snippet': snippet,
            'content': full_content,
            'date': pub_date,
            'date_confidence': confidence
        })
        print(f"Added deferred result for: {url}")

    def _research_func(self, query: str) -> str:
        try:
            print("Starting research function with query:", query)
            # Reset global counters and accumulators at the start of each search
            self.total_image_interpretations = 0
            self.accumulated_extracted_texts = []
            self.results = []  # Initialize results

            refined_query, required_terms, excluded_terms, time_range = self._refine_query(query)
            print("After refine_query")
            
            results = self._google_search(refined_query, required_terms, excluded_terms, time_range)
            print(f"After google_search, got {len(results)} results")
            
            
            structured_data = self._post_process(results, query)
            return structured_data

        except Exception as e:
            print(f"Error in research function: {e}")
            print(f"Full error details: {traceback.format_exc()}")
            return f"Error performing research: {str(e)}"


    def __del__(self):
        """Clean up the Selenium driver"""
        if hasattr(self, 'driver'):
            try:
                self.driver.quit()
            except:
                pass

    def _refine_query(self, query: str) -> tuple:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        prompt = (
            f"Given the following query, please provide:\n"
            f"1. A refined search query optimized for Google Search that will return the most relevant and recent results\n"
            f"2. Required terms that MUST appear in results to ensure relevancy\n"
            f"3. Excluded terms to filter out irrelevant content\n"
            f"4. A time range for the search based on the query's temporal needs\n"
            f"Current date is {current_date}\n\n"
            f"Original Query: {query}\n\n"
            f"Consider:\n"
            f"- If the query asks about future events/predictions, include relevant year\n"
            f"- If the query is about current events, prioritize recency\n"
            f"- Identify key subject terms that must be present\n"
            f"- Identify terms that would indicate off-topic content\n\n"
            f"Respond in the following JSON format:\n"
            "{\n"
            '    "refined_query": "your refined query here",\n'
            '    "required_terms": ["term1", "term2"],\n'
            '    "excluded_terms": ["term1", "term2"],\n'
            '    "time_range": "recent/week/month/year/any"\n'
            "}"
        )
        
        response = openai_llm.invoke([HumanMessage(content=prompt)])
        try:
            result = json.loads(clean_response(response.content.strip()))
            return (
                result["refined_query"],
                result["required_terms"],
                result["excluded_terms"],
                result["time_range"]
            )
        except json.JSONDecodeError:
            return query, [], [], "any"

    def _clean_results(self, results):
        cleaned_results = []
        for result in results:
            content = result['content']
            # Remove incomplete sentences
            sentences = content.split('.')
            complete_sentences = [s.strip() + '.' for s in sentences if len(s.split()) > 5]
            cleaned_content = ' '.join(complete_sentences)
            
            # Include publication date and URL in cleaned results
            cleaned_results.append({
                'url': result['url'],
                'content': cleaned_content,
                'date': result.get('date'),
                'date_confidence':result.get('date_confidence',''),
                'title': result.get('title', '')
            })
        return cleaned_results

    def _standardize_date(self, date_str: str, current_date: datetime.datetime) -> tuple:
        if not date_str:
            return None, 'low'
            
        prompt = (
            f"Convert this date string to a standardized format.\n"
            f"Current date: {current_date.strftime('%Y-%m-%d')}\n"
            f"Input date: {date_str}\n\n"
            f"Return the result in this JSON format:\n"
            "{\n"
            '    "standardized_date": "YYYY-MM-DD",\n'  # Use null if can't determine
            '    "confidence": "high/medium/low"\n'
            "}\n\n"
            f"Handle relative dates (e.g., '2 days ago', 'last week') using the current date.\n"
            f"If you cannot determine a specific date, return null for standardized_date."
        )
        
        try:
            response = openai_llm_mini.invoke([HumanMessage(content=prompt)])
            result = json.loads(clean_response(response.content.strip()))
            return result.get('standardized_date'), result.get('confidence', 'low')
        except Exception as e:
            print(f"Error standardizing date: {e}")
            return None, 'low'

    def _post_process(self, json_results: str, query: str) -> str:
        try:

            cleaned_results = self._clean_results(json_results)
             # Sort results by date confidence and actual date
            current_date = datetime.datetime.now()
            for result in cleaned_results:
                std_date, confidence = self._standardize_date(result.get('date'), current_date)
                result['standardized_date'] = std_date
                result['date_confidence'] = confidence
                
                if std_date:
                    try:
                        date = datetime.datetime.strptime(std_date, '%Y-%m-%d')
                        result['age_days'] = (current_date - date).days
                    except (ValueError, TypeError):
                        result['age_days'] = float('inf')
                else:
                    result['age_days'] = float('inf')

            # Sort by confidence first, then by age
            cleaned_results.sort(
                key=lambda x: (
                    0 if x['date_confidence'] == 'high' else 1 if x['date_confidence'] == 'medium' else 2,
                    x['age_days']
                )
            )

            # Determine temporal requirements based on query
            temporal_prompt = (
                f"Analyze this query and determine its temporal requirements: '{query}'\n"
                f"Respond ONLY in JSON format:\n"
                "{\n"
                '    "recency_importance": "critical/high/medium/low",\n'
                '    "max_age_days": number,\n'
                '    "requires_future_prediction": true/false\n'
                "}"
            )
            
            temporal_response = openai_llm_mini.invoke([HumanMessage(content=temporal_prompt)])
            temporal_requirements = json.loads(clean_response(temporal_response.content.strip()))

            # Sort and filter results based on temporal requirements
            current_date = datetime.datetime.now()
            processed_results = []
            
            for result in cleaned_results:
                if result['age_days'] != float('inf'):
                    if temporal_requirements['recency_importance'] == 'critical' and result['age_days'] > 30:
                        continue
                    elif temporal_requirements['recency_importance'] == 'high' and result['age_days'] > 90:
                        continue
                    elif temporal_requirements['max_age_days'] and result['age_days'] > temporal_requirements['max_age_days']:
                        continue
                
                processed_results.append(result)


            # Prepare the data for final processing
            
        # Prepare content data with standardized dates
            content_data = "\n\n".join([
                f"Source: {item['url']}\n"
                f"Date: {item.get('standardized_date', 'Unknown')} (Confidence: {item['date_confidence']})\n"
                f"Content: {item['content']}"
                for item in processed_results
            ])

            final_prompt = (
                f"Analyze and synthesize the following search results for the query: '{query}'\n\n"
                f"Temporal requirements:\n"
                f"- Recency importance: {temporal_requirements['recency_importance']}\n"
                f"- Requires future prediction: {temporal_requirements['requires_future_prediction']}\n\n"
                f"Instructions:\n"
                f"1. Synthesize the information into a comprehensive response that directly addresses the query\n"
                f"2. When information conflicts, prefer more recent sources\n"
                f"3. For predictive queries, base conclusions primarily on the most recent trends and data\n"
                f"4. For historical or scientific queries, include relevant information across time periods\n"
                f"5. Include temporal context when it adds value (e.g., 'As of November 2023...' or 'Research from 2022 showed...')\n"
                f"6. Highlight any significant changes or trends over time\n"
                f"7. Note any important temporal gaps or limitations in the data\n\n"
                f"Data:\n{content_data}\n\n"
                f"Provide a well-structured response that maintains the detail and nuance from the sources while "
                f"organizing the information in a clear and logical way."
            )

            response = openai_llm_mini.invoke([HumanMessage(content=final_prompt)])
            return response.content.strip()

        except Exception as e:
            return f"Error during post-processing: {str(e)}" 

    def _extract_date_with_gpt(self, url: str, title: str, snippet: str, content: str, current_date: datetime.datetime) -> str:
        try:
            # Take the first 1000 characters of content to keep prompt size reasonable
            content_preview = content[:1000] + ("..." if len(content) > 1000 else "")
            
            prompt = (
                f"Given the following information about an article, determine its publication date.\n"
                f"Current date: {current_date.strftime('%Y-%m-%d')}\n"
                f"URL: {url}\n"
                f"Title: {title}\n"
                f"Snippet: {snippet}\n"
                f"Article content preview: {content_preview}\n\n"
                f"Look for:\n"
                f"1. Explicit dates in the content\n"
                f"2. Relative time references (e.g., '4 days ago', 'last week')\n"
                f"3. References to current events or sports seasons\n"
                f"4. Date-specific context (e.g., 'this season', 'upcoming game this Sunday')\n\n"
                f"Return your best estimate of the publication date in this JSON format:\n"
                "{\n"
                '    "date": "YYYY-MM-DD",\n'
                '    "confidence": "high/medium/low",\n'
                '    "reasoning": "brief explanation of how you determined the date"\n'
                "}\n\n"
                f"If you cannot determine a date, use null for the date value. "
                f"Ensure your response is ONLY the JSON object, nothing else."
            )

            response = openai_llm_mini.invoke([HumanMessage(content=prompt)])
            response_text = clean_response(response.content.strip())
            
            # Additional cleaning to ensure we have valid JSON
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON response: {response_text}")
                print(f"JSON error: {e}")
                # Attempt to fix common JSON issues
                response_text = response_text.replace("'", '"')  # Replace single quotes with double quotes
                response_text = re.sub(r'(\w+):', r'"\1":', response_text)  # Add quotes around keys
                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError:
                    print("Failed to parse JSON even after cleaning")
                    return None, 'low'
            
            date = result.get('date')
            confidence = result.get('confidence', 'low')
            
            # Validate the date format
            if date:
                try:
                    datetime.datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    print(f"Invalid date format received: {date}")
                    return None, 'low'
            
            return date, confidence

        except Exception as e:
            print(f"Error extracting date with GPT: {str(e)}")
            print(f"Full response: {response.content if 'response' in locals() else 'No response'}")
            return None, 'low'
    def _research_func(self, query: str) -> str:
        try:
            print("Starting research function with query:", query)
            # Reset global counters and accumulators at the start of each search
            self.total_image_interpretations = 0
            self.accumulated_extracted_texts = []
            refined_query, required_terms, excluded_terms, time_range = self._refine_query(query)
            print("After refine_query")
            
            results = self._google_search(refined_query, required_terms, excluded_terms, time_range)
            print(f"After google_search, got {len(results)} results")
            
           
            try:  # Add try-except here

                structured_data = self._post_process(results, query)
                return structured_data
            except Exception as e:
                print(f"Error in post-processing: {e}")
                return json.dumps(results, indent=True)  # Return raw results if post-processing fails

        except Exception as e:
            print(f"Error in research function: {e}")
            print(f"Full error details: {traceback.format_exc()}")  # Add this for more detail
            return f"Error performing research: {str(e)}"
        

# Base Agent class to handle common functionality
class BaseAgent:
    def __init__(self, name: str, llm, role: str, available_tools: List[Tool]):
        self.name = name
        self.llm = llm
        self.role = role
        self.available_tools = available_tools
        self.system_message = self._construct_system_message()

    def _construct_system_message(self) -> str:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        base_message = f"You are {self.name}, {self.role}\n\n"
        base_message += f"Today's date is {current_date}.\n"
        if self.available_tools:
            base_message += "Available tools:\n"
            for tool in self.available_tools:
                base_message += f"- {tool.name}: {tool.description}\n"
        '''
        base_message += "\nYou must respond with ONLY a JSON object containing:\n"
        base_message += "- 'thought': your reasoning process\n"
        base_message += "- 'response': your response to the conversation which should be in narrative form (not nested JSON) \n"
        base_message += "- 'tool_calls': (optional, use tools if response would benefit materially from current info) list of tools to invoke, each with 'tool_name' and 'tool_input'\n"
        base_message +="Reminder, response ONLY with JSON. \n"
        return base_message
        '''
        base_message += "\nYou must respond with ONLY a valid JSON object using the following strict formatting rules:\n"
        base_message += "1. Use double quotes (not single quotes) for all JSON keys and string values\n"
        base_message += "2. For newlines in strings, use the escaped sequence '\\n' (not actual line breaks)\n"
        base_message += "3. The JSON object must contain these exact keys:\n"
        base_message += "   - 'thought': your reasoning process as a single-line string\n"
        base_message += "   - 'response': your response as a single-line string (use '\\n' for paragraph breaks)\n"
        base_message += "   - 'tool_calls': (optional) array of objects with 'tool_name' and 'tool_input' strings\n"
        base_message += "4. Do not include any markdown formatting or code blocks\n"
        base_message += "5. Do not wrap the JSON in quotes or backticks\n"
        base_message += "Example format:\n"
        base_message += '{"thought":"reasoning here","response":"first paragraph\\n\\nsecond paragraph","tool_calls":[]}\n'
        base_message += "Ensure your response is exactly one JSON object that can be parsed by JSON.parse()\n"
        return base_message

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
    def __init__(self, name: str, llm, role: str, topology: str, agent_names: List[str], available_tools: List[Tool]):
        super().__init__(name, llm, role, available_tools)
        self.topology = topology
        self.agent_names = agent_names
        if topology == 'last_decides_next':
            self.system_message += "\n- 'next_agent': the name of the next agent to speak"

    def _process_response(self, response_text: str, state: AgentState, retry_count: int = 0) -> Tuple[Dict, List[Dict]]:
        MAX_RETRIES = 3
        response_data, tool_results = super()._process_response(response_text, state, retry_count)
        if 'error' in response_data:
            return response_data, tool_results
        if 'response' not in response_data:
            if retry_count < MAX_RETRIES:
                return self._process_response(response_text, state, retry_count + 1)
            else:
                return {"error": f"Error: No 'response' key found after {MAX_RETRIES} attempts"}, tool_results
        return response_data, tool_results

    def generate_response(self, state: AgentState) -> AgentState:
        messages = [SystemMessage(content=self.system_message)] + state['conversation_messages']
        response = self.llm.invoke(messages)
        
        # Process response and handle any tool calls
        response_data, tool_results = self._process_response(response.content, state)
        
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

        # Add final response to conversation
        response_message = f"{self.name}: {response_data['response']}"
        state['conversation_messages'].append(HumanMessage(content=response_message))

        # Handle next agent selection based on topology
        if self.topology == 'last_decides_next':
            next_agent = response_data.get('next_agent')
            if next_agent and next_agent in [agent for agent in self.agent_names]:
                state['saved_next_agent'] = next_agent
            else:
                state['nextAgent'] = 'END'
        elif 'moderator' in self.topology.lower():
            state['nextAgent']="Moderator"
        
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
        
        # Process response and handle any tool calls
        response_data, tool_results = self._process_response(response.content, state)
        
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

        if state['nextAgent'].upper() == 'END' and response_data.get('final_thoughts'):
            state['conversation_messages'].append(
                AIMessage(content=f"{self.name}: {response_data['final_thoughts']}")
            )
            state['nextAgent'] = 'END'
        else:
            state['conversation_messages'].append(
                HumanMessage(content=f"{self.name}: {response_data['agent_instruction']}")
            )
            state['nextAgent'] = response_data.get('next_agent', 'END')
        
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
            f"2. 'agents': A list of agents where:\n"
            f"   - If topology_type is 'moderator_discretionary' or 'moderated_round_robin', the first agent MUST be the moderator agent with:\n"
            f"     * 'name': 'Moderator'\n"
            f"     * 'type': The LLM to use for the moderator\n"
            f"     * 'role': A system prompt describing the moderator's role\n"
            f"   - For all other agents (and all agents in non-moderated topologies):\n"
            f"     * 'name': The agent's friendly name\n"
            f"     * 'type': The LLM to use (from the available agents)\n"
            f"     * 'role': The system prompt for the agent that describes who the agent is\n"
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
            f"Generate a proposed system prompt for the moderator, which includes the goal for the discussion, the goal for feedback agent participation order.\n\n"
            f"Please return the moderator prompt without any additional text or formatting."
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        moderator_prompt = clean_response(response.content)
        return moderator_prompt


# Initialize the SetupAgent
setup_agent = SetupAgent(openai_llm)
# Moderator agent will be initialized after setup

def should_continue(state: AgentState) -> str:
    nextState = state.get('nextAgent', 'END')
    if nextState.upper() =='END':
        nextState = END
    return nextState

def format_messages(messages: List[Union[SystemMessage, HumanMessage, AIMessage]], suppressToolMsg = True) -> str:
    if suppressToolMsg:
        return "\n\n".join([f"{m.content}" for m in messages if not m.content.startswith(TOOLPREFIX)])
    else:
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
        report_instructions = gr.Textbox(label="Prompt for reporter", value = reporterPrompt,lines=2)

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
            maximum=20,
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
