import os
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
else:
    # Initialize the Gemini Client
    client = genai.Client(api_key=api_key)

    print("Sending request to Gemini 3 Flash Preview...")
    try:
        # Using the model name found in the listing: gemini-3-flash-preview
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents="Say hello and confirm you are the Gemini 3 Flash model."
        )
        
        print("\n--- Model Response ---")
        print(response.text)
        print("----------------------")
        
    except Exception as e:
        print(f"\nError: {e}")
