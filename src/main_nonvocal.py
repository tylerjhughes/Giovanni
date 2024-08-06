from Giuseppe import giuseppe

tutor = giuseppe()

while True:
    try:
        user_message = input("You: ")
    except:
        tutor.text2voice("Sorry, I didn't quite get that.")
        continue

    # Check if the user wants to conclude the conversation
    if user_message.lower().strip().strip('.') == "conclude":
        print("Giuseppe: Goodbye.")
        break

    response = tutor.chat_gpt_api(user_message)
    
    try:
        print("Giuseppe: " + response.choices[0].message.content)
    except:
        print("Giuseppe: I'm lost for words.")


