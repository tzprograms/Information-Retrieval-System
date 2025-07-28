from google.generativeai import Agent
from tools.google_tool import GoogleSearchTool
from tools.bing_tool import BingSearchTool
from tools.yahoo_tool import YahooSearchTool

def build_agent(api_key, cse_id):
    tools = [
        GoogleSearchTool(api_key, cse_id),
        BingSearchTool(),
        YahooSearchTool()
    ]
    return Agent(tools=tools)
