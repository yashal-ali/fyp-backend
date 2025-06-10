import requests

class OllamaTextGenerator:
    def __init__(self, model="llama3", temperature=0.7):
        self.model = model
        self.temperature = temperature

    def generate(self, messages, config=None):
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": self.temperature,
            }
        )

        data = response.json()
        return {"text": [data["response"]]}
