from google.generativeai import Tool

class BingSearchTool(Tool):
    name = "bing_search"
    description = "Use this to search Bing and get web results."

    def run(self, input_text: str) -> str:
        # Placeholder - replace with actual Bing API
        return "\n".join([f"Bing result for: {input_text}"] * 3)
