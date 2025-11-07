import os
import sys
import warnings
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from typing import Optional, List
import json

import google.generativeai as genai
from google.api_core.exceptions import NotFound


class GeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        self.model_name = model_name
        self._configure(api_key)
        self.model = genai.GenerativeModel(self.model_name)

    @staticmethod
    def _configure(api_key: str) -> None:
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY", "")
        genai.configure(api_key=api_key)

    def _retry_models(self) -> List[str]:
        base = self.model_name
        candidates = list(dict.fromkeys([
            base
        ]))
        return candidates

    def generate(self, prompt: str, temperature: float = 0.4, max_output_tokens: Optional[int] = 2000) -> str:
        last_err: Optional[Exception] = None
        for name in self._retry_models():
            try:
                self.model = genai.GenerativeModel(name)
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                    },
                )
                # Robust text extraction even if response.text raises
                try:
                    text = response.text  # quick accessor
                except Exception:
                    text = ""
                    try:
                        if getattr(response, "candidates", None):
                            parts = getattr(response.candidates[0].content, "parts", [])
                            text = " ".join([getattr(p, "text", "") for p in parts if getattr(p, "text", "")])
                    except Exception:
                        text = ""
                if text:
                    return text
                # If model replied but empty/blocked, continue to next candidate model
                continue
            except NotFound as e:
                last_err = e
                continue
        if last_err:
            raise last_err
        return ""

    def extract_json(self, prompt: str, temperature: float = 0.0, max_output_tokens: Optional[int] = 2000) -> dict:
        """
        Generate a response and parse it as JSON. Returns an empty dict if parsing fails.
        """
        text = self.generate(prompt, temperature=temperature, max_output_tokens=max_output_tokens)
        if not text:
            return {}
        
        # Clean up the text - remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```"):
            # Remove ```json or ``` markers
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        
        # Try to parse JSON
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, try to extract JSON object from the text
            # Look for {...} pattern with better nested handling
            # Try to find the first { and matching }
            start = text.find('{')
            if start != -1:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[start:i+1])
                            except (json.JSONDecodeError, ValueError):
                                break
            return {}


