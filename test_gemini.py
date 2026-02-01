from google import genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from .env
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
else:
    print(f"Using API key: {api_key[:10]}... (length: {len(api_key)})")

    client = genai.Client(api_key=api_key)

    print("\nAttempting with 'gemini-2.0-flash'...")
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents="Automotive test engineer: Give me a brief, 1-sentence description of a 'Low Temperature Operation Test'."
        )
        print(f"Success! Response: {response.text}")
    except Exception as e:
        print(f"Error with gemini-2.0-flash: {e}")
