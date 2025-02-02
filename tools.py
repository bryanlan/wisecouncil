# tools.py

from typing import List, Dict, Any, Tuple, Optional
import time
import base64
import os
import io
import json
import datetime
import requests
import re
import traceback
import copy
from PIL import Image
from urllib.parse import quote_plus
from pathlib import Path
from io import BytesIO

from googleapiclient.discovery import build
from readability import Document
import justext
import html2text
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from utils import clean_response, clean_extracted_text
from config import keys, TOKEN_LIMIT_FOR_SUMMARY
from llms import sota_llm, cheap_llm, HumanMessage

class Tool:
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func

    def invoke(self, input_data: str) -> str:
        """
        Main method to be called by the Agents or Moderator. 
        It delegates to the assigned 'func'.
        """
        return self.func(input_data)

class ResearchTool(Tool):
    def __init__(self, max_total_image_interpretations: int = 10, max_interpretations_per_page: int = 4, dismissPopups=True):
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
        self.MIN_RESULTS = 3  # Minimum number of results to process
        self.MAX_RESULTS = 10  # Maximum number of results to process
        
        # Common problematic domains that might block scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
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
        self.has_successful_search = False  # Add this line

        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.research_counter = 0

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

    def _dismiss_popups(self):
        """
        Attempts to dismiss common popups/dialogs on the current page.
        """
        max_attempts = 3
        wait_time = 5
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
                        continue

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
                        continue

                # Optional: Remove popups via JS
                self._remove_popups_via_js()

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

    def _research_func(self, query: str) -> str:
        try:
            # Increment the research counter at the start of each new research
            self.research_counter += 1
            print(f"Starting research function with query: {query}")
            # Reset global counters and accumulators
            self.total_image_interpretations = 0
            self.accumulated_extracted_texts = []
            self.results = []

            refined_query, required_terms, excluded_terms, time_range = self._refine_query(query)
            print("After refine_query")
            
            results = self._google_search(refined_query, required_terms, excluded_terms, time_range)
            print(f"After google_search, got {len(results)} results")
            
            if not results:
                return "No relevant information found for the given query."
            
            structured_data = self._post_process(results, query)
            return structured_data

        except Exception as e:
            print(f"Error in research function: {e}")
            print(f"Full error details: {traceback.format_exc()}")
            return f"Error performing research: {str(e)}"

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
        
        response = sota_llm.invoke([HumanMessage(content=prompt)])
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

    def _resize_image_for_api(self, img: Image.Image, max_size: int = 1024) -> Image.Image:
        """
        Resize image to be within API limits while maintaining aspect ratio.
        """
        # Get current dimensions
        width, height = img.size
        
        # Calculate scaling factor
        scale = min(max_size / width, max_size / height)
        
        # Only resize if image is too large
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        # Convert to RGB if necessary (removing alpha channel)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            bg.paste(img, mask=img.split()[-1])
            img = bg
            
        return img

    def _detect_captcha(self) -> Tuple[bool, Optional[str]]:
        """
        Takes a screenshot of the current page and uses cheap_llm to detect if it's a CAPTCHA.
        Returns a tuple of (is_captcha, captcha_url).
        """
        try:
            # Take screenshot
            screenshot = self.driver.get_screenshot_as_png()
            img = Image.open(BytesIO(screenshot))
            
            # Resize image
            img = self._resize_image_for_api(img)
            
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Analyze screenshot with cheap_llm
            prompt = (
                "Analyze this screenshot and determine if it shows a Google CAPTCHA verification page.\n"
                "If it is a CAPTCHA page:\n"
                "1. Look for any URL or link that needs to be clicked to complete verification\n"
                "2. Return the exact URL if found\n\n"
                "Respond in JSON format:\n"
                "{\n"
                '    "is_captcha": true/false,\n'
                '    "captcha_url": "url or null",\n'
                '    "explanation": "brief explanation of what you see"\n'
                "}"
            )

            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    }
                ])
            ]
            
            response = cheap_llm.invoke(messages)
            result = json.loads(clean_response(response.content))
            
            return result["is_captcha"], result.get("captcha_url")
            
        except Exception as e:
            print(f"Error in CAPTCHA detection: {e}")
            return False, None

    def _analyze_image(self, image_data: str, query: str) -> Optional[str]:
        """
        Analyze an image using cheap_llm's multimodal capabilities.
        Returns a description of the image content relevant to the query context.
        
        Args:
            image_data (str): Base64 encoded image data
            query (str): The search query to provide context
        """
        try:
            # Save image for debugging
            try:
                img_data = base64.b64decode(image_data)
                img = Image.open(BytesIO(img_data))
                
                # Resize image
                img = self._resize_image_for_api(img)
                
                # Save resized image
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                image_data = base64.b64encode(buffered.getvalue()).decode()
                
                with open('redditpic.jpg', 'wb') as f:
                    f.write(buffered.getvalue())
                print("Saved debug image as redditpic.jpg")
            except Exception as e:
                print(f"Failed to save debug image: {e}")

            prompt = (
                f"Analyze this image in the context of the following query: '{query}'\n\n"
                f"Please describe:\n"
                f"1. Any information in the image that's relevant to the query\n"
                f"2. Any visible text that relates to the topic\n"
                f"3. If it's a chart/graph, enumerate all key data points and explain the data especially as it relates to the query\n"
                f"4. If it's a meme, explain its meaning in relation to the topic\n\n"
                f"Do not leave out any key points or data, add as much detail as necessary\n"
                f"Focus only on aspects relevant to '{query}'. Ignore unrelated content. If all content is unrelated, return 'No relevant content found in image'."
            )

            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ])
            ]
            
            response = cheap_llm.invoke(messages)
            if response.content.strip() == "No relevant content found in image.":
                return ""
            else: 
                return response.content.strip()
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None

    def _google_search(self, query: str, required_terms: list, excluded_terms: list, time_range: str, num_results: int = None):
        print(f"Starting google_search with query: {query}")
        all_results = []
        processed_urls = set()
        page_count = 0
        max_pages = 3
        
        target_results = num_results if num_results is not None else self.MAX_RESULTS
        
        try:
            search_query = f'{query} -site:youtube.com -site:youtu.be -site:washingtonpost.com'
            url = f"https://cse.google.com/cse?cx={self.search_engine_id}&q={quote_plus(search_query)}"
            
            # Function to perform the search
            def perform_search():
                self.driver.get(url)
                time.sleep(2)  # Wait for page to load
                
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "gsc-result"))
                )
                time.sleep(3)

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
                
                return self.driver.execute_script(js_script)

            # Initial search attempt
            page_results = perform_search()
            print(f"Found {len(page_results)} results on page {page_count + 1}")

            # If no results, check for CAPTCHA
            if not page_results:
                print(f"No results found, checking for CAPTCHA...")
                is_captcha, captcha_url = self._detect_captcha()
                if is_captcha:
                    # Create a dialog message with the URL
                    dialog_msg = f"Google CAPTCHA detected. Please complete the verification at:\n{captcha_url if captcha_url else 'URL not found'}"
                    print(dialog_msg)
                    # Wait for user to complete CAPTCHA
                    input("Press Enter after completing the CAPTCHA verification...")
                    # Retry the search
                    print("Retrying search after CAPTCHA verification...")
                    page_results = perform_search()
                    if not page_results:
                        print(f"Still no results found after CAPTCHA verification")
                        return []
            
            while page_count < max_pages:
                if not page_results:
                    print(f"No results found for URL: {url}")
                    return []

                # Filter out duplicates or videos
                page_results = [
                    result for result in page_results
                    if result['url'] not in processed_urls
                    and result['url'].startswith('http')
                    and not any(domain in result['url'].lower() for domain in [
                        'youtube.com', 'youtu.be', 'vimeo.com',
                        'dailymotion.com', 'tiktok.com'
                    ])
                ]

                if page_results:
                    highly_relevant, maybe_relevant = self._evaluate_search_results(page_results, query)
                    
                    for result in highly_relevant:
                        if len(all_results) >= target_results:
                            break
                        try:
                            url = result['url']
                            processed_urls.add(url)
                            
                            print(f"Processing highly relevant URL: {url}")
                            full_content, pub_date, confidence, image_reading_needed = self._extract_content(
                                url, result['title'], result['snippet'], query
                            )
                            if full_content and len(full_content.split()) > 50:
                                all_results.append({
                                    'url': url,
                                    'title': result['title'],
                                    'snippet': result['snippet'],
                                    'content': full_content,
                                    'date': pub_date,
                                    'date_confidence': confidence
                                })
                                print(f"Added highly relevant result: {url}")
                        except Exception as e:
                            print(f"Error processing result: {e}")
                            continue

                    if len(all_results) < self.MIN_RESULTS:
                        for result in maybe_relevant:
                            if len(all_results) >= target_results:
                                break
                            try:
                                url = result['url']
                                processed_urls.add(url)
                                
                                print(f"Processing maybe relevant URL: {url}")
                                full_content, pub_date, confidence, image_reading_needed = self._extract_content(
                                    url, result['title'], result['snippet'], query
                                )
                                if full_content and len(full_content.split()) > 50:
                                    all_results.append({
                                        'url': url,
                                        'title': result['title'],
                                        'snippet': result['snippet'],
                                        'content': full_content,
                                        'date': pub_date,
                                        'date_confidence': confidence
                                    })
                                    print(f"Added maybe relevant result: {url}")
                            except Exception as e:
                                print(f"Error processing result: {e}")
                                continue

                if len(all_results) >= self.MIN_RESULTS and (
                    len(all_results) >= target_results or 
                    not self._should_load_more_results(all_results, query)
                ):
                    break

                if not self._click_next_page():
                    break
                    
                page_count += 1
                # Get results from next page
                page_results = perform_search()
                print(f"Found {len(page_results)} results on page {page_count + 1}")

            if len(all_results) > 0:
                self.has_successful_search = True
            return all_results

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def _evaluate_search_results(self, search_results: List[Dict], query: str) -> Tuple[List[Dict], List[Dict]]:
        if not search_results:
            return [], []
        results_text = "\n\n".join([
            f"Result {i+1}:\n"
            f"Title: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Snippet: {result['snippet']}"
            for i, result in enumerate(search_results)
        ])

        prompt = (
            f"Given this search query: '{query}'\n\n"
            f"Evaluate these search results and classify them into two categories:\n"
            f"1. Highly relevant\n"
            f"2. Maybe relevant\n\n"
            f"Search Results:\n{results_text}\n\n"
            f"Respond in JSON with:\n"
            "{\n"
            '   "highly_relevant": [result_numbers],\n'
            '   "maybe_relevant": [result_numbers],\n'
            '   "reasoning": "some justification"\n'
            "}"
        )

        try:
            response = cheap_llm.invoke([HumanMessage(content=prompt)])
            result = json.loads(clean_response(response.content))
            
            highly_relevant = [search_results[i-1] for i in result['highly_relevant']]
            maybe_relevant = [search_results[i-1] for i in result['maybe_relevant']]
            
            print(f"Evaluation complete: {len(highly_relevant)} highly relevant, {len(maybe_relevant)} maybe relevant")
            print(f"Reasoning: {result['reasoning']}")
            return highly_relevant, maybe_relevant
        except Exception as e:
            print(f"Error evaluating search results: {e}")
            return search_results, []

    def _should_load_more_results(self, current_results: List[Dict], query: str) -> bool:
        if len(current_results) >= self.MAX_RESULTS:
            return False
        if len(current_results) < self.MIN_RESULTS:
            return True

        results_summary = "\n".join([
            f"- {result.get('title', 'Untitled')} ({result.get('date', 'Unknown date')})"
            for result in current_results
        ])

        prompt = (
            f"Given this search query: '{query}'\n\n"
            f"Current results collected:\n{results_summary}\n\n"
            f"Should we load more search results? Return 'yes' or 'no'."
        )

        try:
            response = cheap_llm.invoke([HumanMessage(content=prompt)])
            decision = response.content.lower().strip()
            print(f"Load more decision: {decision}")
            return decision.startswith('yes')
        except Exception as e:
            print(f"Error in load more decision: {e}")
            return len(current_results) < 5

    def _click_next_page(self) -> bool:
        try:
            # Wait for results to load first
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "gsc-results"))
            )
            time.sleep(3)  # Increased delay to ensure page is fully loaded
            
            # First try to find the current page number
            try:
                current_page = self.driver.find_element(By.CSS_SELECTOR, ".gsc-cursor-current-page")
                current_page_num = int(current_page.text)
                print(f"Current page: {current_page_num}")
            except Exception as e:
                print(f"Could not determine current page: {e}")
                current_page_num = 1

            # Find all page buttons
            try:
                # Use a more general selector to find all page numbers
                page_buttons = self.driver.find_elements(By.CSS_SELECTOR, ".gsc-cursor-page")
                
                if not page_buttons:
                    print("No pagination buttons found")
                    return False
                    
                print(f"Found {len(page_buttons)} pagination buttons")
                
                # Find the next page button
                next_page_button = None
                for button in page_buttons:
                    try:
                        button_num = int(button.text)
                        if button_num == current_page_num + 1:
                            next_page_button = button
                            break
                    except ValueError:
                        continue
                    
                if not next_page_button:
                    print("Could not find next page button")
                    return False
                    
                # Ensure the button is in view
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center', inline: 'center'});", 
                    next_page_button
                )
                time.sleep(1)
                
                # Try to click using different methods
                try:
                    # Method 1: Direct click with retry
                    max_retries = 3
                    for _ in range(max_retries):
                        try:
                            next_page_button.click()
                            break
                        except:
                            time.sleep(1)
                            continue
                            
                    # Method 2: JavaScript click if Method 1 failed
                    if not self._verify_page_change(current_page_num):
                        self.driver.execute_script("arguments[0].click();", next_page_button)
                        
                    # Method 3: Action chains if Method 2 failed
                    if not self._verify_page_change(current_page_num):
                        from selenium.webdriver.common.action_chains import ActionChains
                        actions = ActionChains(self.driver)
                        actions.move_to_element(next_page_button)
                        actions.click()
                        actions.perform()
                    
                    # Final verification
                    if self._verify_page_change(current_page_num):
                        print("Successfully changed page")
                        return True
                        
                    print("Failed to verify page change after all click attempts")
                    return False
                    
                except Exception as click_error:
                    print(f"Error clicking next page button: {click_error}")
                    return False
                    
            except Exception as e:
                print(f"Error finding pagination buttons: {e}")
                return False
                
        except Exception as e:
            print(f"Error in _click_next_page: {e}")
            traceback.print_exc()
            return False

    def _verify_page_change(self, previous_page_num: int, max_wait: int = 5) -> bool:
        """
        Verify that the page has actually changed after clicking next.
        Returns True if page change is confirmed, False otherwise.
        """
        try:
            start_time = time.time()
            while time.time() - start_time < max_wait:
                try:
                    current_page = self.driver.find_element(By.CSS_SELECTOR, ".gsc-cursor-current-page")
                    current_num = int(current_page.text)
                    if current_num > previous_page_num:
                        print(f"Page change verified: {previous_page_num} -> {current_num}")
                        return True
                except:
                    pass
                time.sleep(0.5)
            return False
        except Exception as e:
            print(f"Error verifying page change: {e}")
            return False

    def _extract_content(self, url: str, title: str, snippet: str, query: str) -> tuple:
        try:
            image_reading_needed = False
            self.driver.get(url)
            print(f"Navigated to URL: {url}")
            time.sleep(2)

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            doc = Document(response.text)
            
            pub_date = None
            confidence = 'low'
            soup = BeautifulSoup(response.text, 'html.parser')
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

            content = doc.summary()
            paragraphs = justext.justext(content, justext.get_stoplist("English"))
            content_parts = []
            for paragraph in paragraphs:
                if not paragraph.is_boilerplate:
                    content_parts.append(paragraph.text)

            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            h.ignore_emphasis = True
            text_content = h.handle(' '.join(content_parts))
            clean_text = clean_extracted_text(text_content)

            if not clean_text.strip():
                print(f"No text content extracted from {url}, will defer image reading.")
                image_reading_needed = True

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
                formatted_text = f"Title: {title}\n"
                if pub_date:
                    formatted_text += f"Date: {pub_date}\n"
                formatted_text += f"Content:\n{clean_text}\n"
                self.accumulated_extracted_texts.append(formatted_text)

            return clean_text, pub_date, confidence, image_reading_needed

        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
            return "", None, 'low', False

    def _extract_date_with_gpt(self, url: str, title: str, snippet: str, content: str, current_date: datetime.datetime) -> str:
        try:
            content_preview = content[:1000] + ("..." if len(content) > 1000 else "")
            prompt = (
                f"Given the following information about an article, determine its publication date.\n"
                f"Current date: {current_date.strftime('%Y-%m-%d')}\n"
                f"URL: {url}\n"
                f"Title: {title}\n"
                f"Snippet: {snippet}\n"
                f"Article content preview: {content_preview}\n\n"
                f"Return your best estimate of the publication date in JSON:\n"
                "{\n"
                '    "date": "YYYY-MM-DD",\n'
                '    "confidence": "high/medium/low"\n'
                "}\n"
                f"ONLY respond with the JSON, no other text."
            )
            response = cheap_llm.invoke([HumanMessage(content=prompt)])
            response_text = clean_response(response.content.strip())
            response_text = response_text.replace("```json", "").replace("```", "").strip()

            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as ex:
                print(f"Invalid JSON response: {response_text}")
                response_text = response_text.replace("'", '"')
                response_text = re.sub(r'(\w+):', r'"\1":', response_text)
                try:
                    result = json.loads(response_text)
                except:
                    return None, 'low'

            date = result.get('date')
            confidence = result.get('confidence', 'low')
            if date:
                try:
                    datetime.datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    return None, 'low'
            return date, confidence

        except Exception as e:
            print(f"Error extracting date with GPT: {str(e)}")
            return None, 'low'

    def _post_process(self, json_results: str, query: str) -> str:
        try:
            cleaned_results = self._clean_results(json_results)

            current_date = datetime.datetime.now()
            for result in cleaned_results:
                std_date, conf = self._standardize_date(result.get('date'), current_date)
                result['standardized_date'] = std_date
                result['date_confidence'] = conf
                if std_date:
                    try:
                        dt = datetime.datetime.strptime(std_date, '%Y-%m-%d')
                        result['age_days'] = (current_date - dt).days
                    except:
                        result['age_days'] = float('inf')
                else:
                    result['age_days'] = float('inf')

            cleaned_results.sort(
                key=lambda x: (
                    0 if x['date_confidence'] == 'high' else 1 if x['date_confidence'] == 'medium' else 2,
                    x['age_days']
                )
            )

            temporal_prompt = (
                f"Analyze this query and determine its temporal requirements: '{query}'\n"
                f"Respond ONLY in JSON format:\n"
                "{\n"
                '    "recency_importance": "critical/high/medium/low",\n'
                '    "max_age_days": number,\n'
                '    "requires_future_prediction": true/false\n'
                "}"
            )
            
            temporal_response = cheap_llm.invoke([HumanMessage(content=temporal_prompt)])
            temporal_requirements = json.loads(clean_response(temporal_response.content.strip()))

            processed_results = []
            summaries = []
            current_content = ""
            current_token_count = 0
            chunk_counter = 0  # Add counter for chunks

            for result in cleaned_results:
                # First apply temporal filtering
                if result['age_days'] != float('inf'):
                    if temporal_requirements['recency_importance'] == 'critical' and result['age_days'] > 30:
                        continue
                    elif temporal_requirements['recency_importance'] == 'high' and result['age_days'] > 90:
                        continue
                    elif temporal_requirements['max_age_days'] and result['age_days'] > temporal_requirements['max_age_days']:
                        continue

                # If result passes temporal filter, process it
                content = (
                    f"Source: {result['url']}\n"
                    f"Date: {result.get('standardized_date', 'Unknown')} "
                    f"(Confidence: {result['date_confidence']})\n"
                    f"Content: {result['content']}\n\n"
                )
                
                # Rough token estimate (1 token ≈ 4 chars)
                content_tokens = len(content) // 4
                
                if current_token_count + content_tokens > TOKEN_LIMIT_FOR_SUMMARY:
                    # Save the current chunk before summarizing
                    chunk_counter += 1
                    self._save_research_chunk(current_content, chunk_counter)
                    
                    # Generate intermediate summary
                    summary = self._generate_intermediate_summary(current_content, query)
                    summaries.append(summary)
                    
                    # Reset accumulator
                    current_content = content
                    current_token_count = content_tokens
                else:
                    current_content += content
                    current_token_count += content_tokens
                
                processed_results.append(result)

            # Handle remaining content
            if current_content:
                chunk_counter += 1
                self._save_research_chunk(current_content, chunk_counter)
                summary = self._generate_intermediate_summary(current_content, query)
                summaries.append(summary)

            # Simply join all summaries with a separator
            return "\n\n=== Next Batch of Results ===\n\n".join(summaries)

        except Exception as e:
            return f"Error during post-processing: {str(e)}"

    def _generate_intermediate_summary(self, content: str, query: str) -> str:
        prompt = (
            f"Analyze and synthesize the following search results for the query: '{query}'\n\n"
            f"Instructions:\n"
            f"1. Synthesize the key information\n"
            f"2. Handle conflicting info carefully, prefer more recent\n"
            f"3. For predictive queries, highlight recent trends\n"
            f"4. Provide summary in plain text\n\n"
            f"Data:\n{content}\n"
        )

        response = cheap_llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _clean_results(self, results):
        cleaned_results = []
        for result in results:
            content = result['content']
            sentences = content.split('.')
            complete_sentences = [s.strip() + '.' for s in sentences if len(s.split()) > 5]
            cleaned_content = ' '.join(complete_sentences)

            cleaned_results.append({
                'url': result['url'],
                'content': cleaned_content,
                'date': result.get('date'),
                'date_confidence': result.get('date_confidence',''),
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
            f"Return ONLY the result in JSON:\n"
            "{\n"
            '    "standardized_date": "YYYY-MM-DD",\n'
            '    "confidence": "high/medium/low"\n'
            "}\n"
        )
        try:
            response = cheap_llm.invoke([HumanMessage(content=prompt)])
            result = json.loads(clean_response(response.content.strip()))
            return result.get('standardized_date'), result.get('confidence', 'low')
        except Exception as e:
            print(f"Error standardizing date: {e}")
            return None, 'low'

    def _save_research_chunk(self, content: str, chunk_number: int) -> None:
        """Save a chunk of research content to a numbered file."""
        try:
            filename = self.log_dir / f"research{self.research_counter}_{chunk_number}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"Error saving research chunk: {e}")
