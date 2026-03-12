import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Listing models with details...")
for m in client.models.list():
    print(f"Name: {m.name}")
    print(f"  Display Name: {getattr(m, 'display_name', 'N/A')}")
    print(f"  Description: {getattr(m, 'description', 'N/A')}")
    print("-" * 20)
