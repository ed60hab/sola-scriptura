import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Looking for a working model...")
for m in client.models.list():
    # In some SDK versions m.supported_methods exists, in others it's m.supported_actions
    methods = getattr(m, 'supported_methods', getattr(m, 'supported_actions', []))
    if 'generateContent' in methods or 'generate_content' in str(methods).lower():
        print(f"Trying {m.name}...")
        try:
            response = client.models.generate_content(
                model=m.name,
                contents=["OK"]
            )
            print(f"SUCCESS with {m.name}!")
            with open("working_model.txt", "w") as f:
                f.write(m.name)
            break
        except Exception as e:
            print(f"FAILED with {m.name}: {e}")
