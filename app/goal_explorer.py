import json
from lida.utils import clean_code_snippet
from lida.datamodel import Goal, TextGenerationConfig, Persona
import re

SYSTEM_INSTRUCTIONS = """
You are an experienced data analyst who can generate insightful GOALS based on a data summary and a user persona. 
Your recommendations MUST:
- Follow visualization best practices (e.g., bar charts > pie charts)
- Use dataset fields from the summary
- Include a question, a visualization (that mentions exact fields), and a rationale.
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A VALID JSON LIST:

```[
  { 
    "index": 0, 
    "question": "What is the distribution of column_name?", 
    "visualization": "histogram of column_name", 
    "rationale": "This tells us how values in column_name are distributed." 
  }
]
```
"""
class GoalExplorer:
    def __init__(self):
        pass

    def extract_json_block(self, text):
        """Extracts JSON array from text using regex."""
        import re
        match = re.search(r"```(?:json)?\s*(\[\s*{.*?}\s*])\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            match = re.search(r"(\[\s*{.*?}\s*])", text, re.DOTALL)
            return match.group(1) if match else None

    def generate(self, summary, textgen_config, text_gen, n=5, persona=None):
        from lida.datamodel import Persona

        if not persona:
            persona = Persona(
                persona="A highly skilled data analyst who can come up with complex, insightful goals about data",
                rationale="")

        user_prompt = f"""The number of GOALS to generate is {n}. The goals should be based on the data summary below:\n\n{summary}\n\n"""
        user_prompt += f"""The generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{persona.persona}' persona.\n"""

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "assistant", "content": f"{user_prompt}\n\n{FORMAT_INSTRUCTIONS}\n\nThe generated {n} goals are:\n"}
        ]

        response = text_gen.generate(messages=messages, config=textgen_config)

        raw_text = response["text"][0]
        json_block = self.extract_json_block(raw_text)


        if not json_block:
            raise ValueError(f"Invalid JSON from model:\n{raw_text}")

        try:
            data = json.loads(json_block)
            if isinstance(data, dict):
                data = [data]
            return [Goal(**x) for x in data]

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing failed: {e}\nExtracted:\n{json_block}")
    def visualize(self, summary, goal, textgen_config: TextGenerationConfig, library="seaborn", return_error: bool = False):
        from lida import Manager
        from app.llama_ollama import LLaMAOllamaLLM

        if isinstance(goal, dict):
            goal = Goal(**goal)

        lida = Manager(text_gen=LLaMAOllamaLLM(model="llama3:8b-instruct-q4_0"))

        try:
            charts = lida.visualize(summary=summary, goal=goal, textgen_config=textgen_config, library=library)
            
            # Ensure you are accessing the correct part of the response
            if isinstance(charts, dict) and 'content' in charts:
                response_content = charts['content']
                # Proceed with your code to process 'response_content'
                return response_content
            else:
                # Handle cases where the expected content is not found
                print(f"Unexpected response structure: {charts}")
                return {"error": "Unexpected response structure"}
            
        except Exception as e:
            print(f"Error during execution of code_specs: {e}")
            if return_error:
                return {"error": str(e)}
            raise
