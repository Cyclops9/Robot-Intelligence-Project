import google.generativeai as genai
import os

# Configure your API key
# You can set this as an environment variable or paste it directly
api_key = os.getenv("GEMINI_API_KEY") 
if not api_key:
    # If not in environment, paste your key here
    api_key = "AIzaSyDceYS8RslgPXDw5E8vqBbEXu6yZjUqtZ0"

genai.configure(api_key=api_key)

print("Fetching available models...\n")
try:
    for m in genai.list_models():
        # Models that support content generation (like Gemini)
        if 'generateContent' in m.supported_generation_methods:
            print(f"Name: {m.name}")
            print(f"  Description: {m.description}")
            print(f"  Input Limit: {m.input_token_limit} tokens")
            print(f"  Output Limit: {m.output_token_limit} tokens")
            print("-" * 40)
except Exception as e:
    print(f"Error fetching models: {e}")