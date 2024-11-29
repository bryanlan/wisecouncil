# tools.py

from typing import List, Dict, Any
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

from langchain_core.messages import HumanMessage
from llms import openai_llm, openai_llm_mini
from utils import clean_response
from config import keys  # Assuming keys is imported from config

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
        
