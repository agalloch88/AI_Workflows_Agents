from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

def generate_x_post(topic: str) -> str:
    # Placeholder for the logic to generate an X post
    prompt = f"""
        You are an expert social media manager,  and you excel at crafting
        viral and highly engaging posts for the X platform (formerly Twitter). 
        
        Your task is to generate a concise and engaging X post which is impactful and 
        tailored to the topic provided by the user. Avoid using hashtags, and avoidlots of emojis 
        (a few are ok, but try to avoid them). 
        
        Keep the post short and focused, ideally under 280 characters, and structure it in a clean, readable way, 
        using line breaks and empty lines to enhance readability.

        Here's the topic provided by the user for the post: <topic>{topic}</topic>
"""
    response = client.responses.create(model="gpt-4o", input=prompt)

    return response.output_text

def main():
    topic = input("What should the post be about? ")

    x_post = generate_x_post(topic)
    print("\nHere is your generated X post:\n")
    print(x_post)


if __name__ == "__main__":
    main()
