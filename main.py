import requests

def generate_x_post(user_input: str) -> str:
    # Placeholder for the logic to generate an X post
    response = requests.post(
        "https://api.example.com/generate_post",
        json={"input": user_input}
    )

def main():
    user_input = input("What should the post be about? ")

    x_post = generate_x_post(user_input)
    print("\nHere is your generated X post:\n")
    print(x_post)


if __name__ == "__main__":
    main()
