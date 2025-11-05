import os

from dotenv import load_dotenv
import google.generativeai as genai

from services.gemini_client import GeminiClient
from config import MODEL_NAME


def list_available_models() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY not set in .env")
        return
    genai.configure(api_key=api_key)
    models = list(genai.list_models())
    print(f"Found {len(models)} models:")
    for m in models:
        name = getattr(m, "name", "?")
        in_types = getattr(m, "input_token_limit", "")
        out_types = getattr(m, "output_token_limit", "")
        sup = getattr(m, "supported_generation_methods", [])
        print(f"- {name} | methods={sup} | in={in_types} out={out_types}")


def quick_generation_test(prompt: str = "Say hello in one sentence.") -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY not set in .env")
        return
    client = GeminiClient(api_key=api_key, model_name=MODEL_NAME)
    print("Using model:", MODEL_NAME)
    out = client.generate(prompt)
    print("Response:\n", out)


if __name__ == "__main__":
    print("=== Listing available models ===")
    list_available_models()
    print("\n=== Quick generation test ===")
    quick_generation_test()


