from Giovanni import giovanni

tutor = giovanni()

while True:
    try:
        user_message = tutor.voice2text()
        print("You: " + user_message)
    except Exception as e:
        print(e)
        tutor.text2voice("Sorry, I didn't quite get that.")
        continue

    if user_message.lower().strip().strip('.') == "conclude":
        print("Giuseppe: Goodbye.")
        tutor.text2voice("Goodbye.")
        break

    response = tutor.chat_gpt_api(user_message)
    print("Giuseppe: " + response.choices[0].message.content)
    tutor.text2voice(response.choices[0].message.content)
