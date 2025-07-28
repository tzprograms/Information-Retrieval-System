from google.generativeai import GenerativeModel

def summarize_text(text: str) -> str:
    model = GenerativeModel("gemini-pro")
    response = model.generate_content(f"Summarize the following information:\n{text}")
    return response.text
