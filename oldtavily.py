class Tool:
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func

    def invoke(self, input_data: str) -> str:
        return self.func(input_data)
    
class ResearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="do_research",
            description="Perform internet research on a given query",
            func=self._research_func
        )
        self.search_tool = TavilySearchResults(
            max_results=7,
            include_answer=True,
            include_raw_content=True,
            include_images=False ,
            search_depth="advanced",
            # include_domains = []
            # exclude_domains = []

        )

    def _refine_query(self, query: str) -> tuple:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        prompt = (
            f"Given the following query, please provide:\n"
            f"1. A refined search query that is specific, clear, and optimized for retrieving relevant information\n"
            f"2. A list of high-quality domains that are likely to have reliable information about this topic\n"
            f"Current date is {current_date}\n\n"
            f"Original Query: {query}\n\n"
            f"Respond in the following JSON format:\n"
            "{\n"
            '    "refined_query": "your refined query here",\n'
            '    "suggested_domains": ["domain1.com", "domain2.org", ...]\n'
            "}"
        )
        
        response = openai_llm.invoke([HumanMessage(content=prompt)])
        try:
            result = json.loads(clean_response(response.content.strip()))
            return result["refined_query"], result["suggested_domains"]
        except json.JSONDecodeError:
            # Fallback in case of parsing error
            return query, []
    def _clean_results(self,results):
        cleaned_results = []
        for result in results:
            content = result['content']
            # Remove incomplete sentences
            sentences = content.split('.')
            complete_sentences = [s.strip() + '.' for s in sentences if len(s.split()) > 5]
            cleaned_content = ' '.join(complete_sentences)
            cleaned_results.append({'url': result['url'], 'content': cleaned_content})
        return cleaned_results

    def _post_process(self, json_results: str, query: str) -> str:
        try:
            # Load the results from the JSON string
            results = json.loads(json_results)
            # Extract content and clean it
            cleaned_results = self._clean_results(results)

            # Prepare the data for the OpenAI model
            content_data = " ".join([item['content'] for item in cleaned_results])
            prompt = (
                f"Here is the data extracted from a search query: '{query}'. "
                f"The data includes a collection of information that needs to be refined and structured for optimal use in further processing. "
                f"Please filter out any irrelevant information and structure the relevant data in a clear and organized way, "
                f"ensuring it is ready for further use. Do not remove or summarize relevant data:\n\n"
                f"Data: {content_data}"
            )

            # Invoke the mini model for post-processing
            response = openai_llm_mini.invoke([HumanMessage(content=prompt)])
            structured_data = response.content.strip()
            return structured_data
        except Exception as e:
            return f"Error during post-processing: {str(e)}"

    def _research_func(self, query: str) -> str:
        try:
            refined_query, suggested_domains = self._refine_query(query)
            
            # Create a new search tool instance with the suggested domains
            search_tool = TavilySearchResults(
                max_results=10,
                include_answer=True,
                include_raw_content=True,
                include_images=False,
                search_depth="advanced",
                include_domains=suggested_domains if suggested_domains else None
            )
            
            results = search_tool.invoke({"query": refined_query})
            json_results = json.dumps(results, indent=True)

            # Post-process the results
            structured_data = self._post_process(json_results, query)
            return structured_data

        except Exception as e:
            return f"Error performing research: {str(e)}"