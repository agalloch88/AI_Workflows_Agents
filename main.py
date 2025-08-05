from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

def generate_x_post(topic: str) -> str:
    # Placeholder for the logic to generate an X post
    with open("post-examples.json", "r") as f:
        examples = json.load(f)

    examples_str = ""
    for i, example in enumerate(examples, 1):
        examples_str += f"""
        <example-{i}>
            <topic>
            {example['topic']}
            </topic>

            <generated-post>
            {example['post']}
            </generated-post>
        </example-{i}>
        """
    prompt = f"""
        You are an expert social media manager,  and you excel at crafting
        viral and highly engaging posts for the X platform (formerly Twitter). 
        
        Your task is to generate a concise and engaging X post which is impactful and 
        tailored to the topic provided by the user. Avoid using hashtags, and avoidlots of emojis 
        (a few are ok, but try to avoid them). 
        
        Keep the post short and focused, ideally under 280 characters, and structure it in a clean, readable way, 
        using line breaks and empty lines to enhance readability.

        Here's the topic provided by the user for the post: <topic>{topic}</topic>

        Here are some examples of topics and generated posts:
        <examples>
            {examples_str}
        </examples>

        Please use the tone, language, structure , and style of the examples provided above to generate a post that is engaging and relevant to the topic provided by the user.
        Don't use the content from the examples!
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
