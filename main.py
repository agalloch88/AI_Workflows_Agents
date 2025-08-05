import requests
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")

def generate_x_post(user_input: str) -> str:
    # Placeholder for the logic to generate an X post
    payload = {
      "model": "gpt-4o",
      "input": user_input,
   }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_KEY}"
        }
    )

def main():
    user_input = input("What should the post be about? ")

    x_post = generate_x_post(user_input)
    print("\nHere is your generated X post:\n")
    print(x_post)


if __name__ == "__main__":
    main()
