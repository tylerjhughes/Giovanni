# Giovanni

Giovanni is an Italian tutor chatbot that can be used to have text based and vocalised conversation to help improve conversational Italian. It uses the [`Whisper`](https://huggingface.co/openai/whisper-base) base model to for speech to text, `ChatGPT-4o` for the responses, and `pydub` for text to speech. An OpenAI API key needs to be used and exported to your environment variables for this to work. Instructions to get a key can be found [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key).

## Usage

Giovanni can either be used in a vocal or non-vocal conversation. Vocal will record an input, convert the recording to text, send the text to ChatGPT and convert the response to speech. Non-vocal is a purely text based interaction.

### Vocal

To run the vocal mode, while in the main directory input the following into the terminal:

```python
python src/main_vocal.py
```

### Non-vocal

To run the vocal mode, while in the main directory input the following into the terminal:

```python
python src/main_nonvocal.py
```

