from google.generativeai import Tool
import requests

class GoogleSearchTool(Tool):
    name = "google_search"
    description = "Use this to search Google and get web results."

    def __init__(self, api_key, cse_id):
        self.api_key = api_key
        self.cse_id = cse_id

    def run(self, input_text: str) -> str:
        url = f"https://www.googleapis.com/customsearch/v1?q={input_text}&key={self.api_key}&cx={self.cse_id}"
        response = requests.get(url)
        items = response.json().get("items", [])
        return "\n".join([item["snippet"] for item in items[:5]])
