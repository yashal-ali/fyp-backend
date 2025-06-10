
import requests

class LLaMAOllamaLLM:
    def __init__(self, model="llama3:8b-instruct-q4_0"):
        self.url = "http://localhost:11434/api/generate"
        self.model = model
        self.provider = "ollama"  # This is needed by LIDA

    def generate(self, messages=None, config=None):
        if isinstance(messages, str):
            prompt = messages  # If it's just a string, use as prompt
        elif isinstance(messages, list):
            prompt = "\n".join([f"{m.get('role', 'user')}: {m['content']}" for m in messages])  # Handle message structure
        else:
            raise ValueError("Invalid 'messages' format passed to generate()")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        # Send the request to the Ollama API
        response = requests.post(self.url, json=payload)
        print("Ollama API Response:", response.status_code, response.text)  # Debug print for response
        response.raise_for_status()  # Will raise HTTPError for bad status codes
        
        # Fix the response structure
        result = response.json().get("response", "")
        print("result----------------------",[result])
        return {"text": [result]}  # Ensure that 'text' is a list, not a dictionary