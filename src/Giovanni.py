from gtts import gTTS
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from pydub import AudioSegment
import sounddevice as sd
import numpy as np
from openai import OpenAI
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
from pydub.playback import play
from io import BytesIO
import json



class giuseppe(object):

    '''
    Class to create an Italian tutor and handle voice-to-text, ChatGPT API, and text-to-speech.
    '''

    def __init__(self):

        self.model = 'gpt-4o-mini'

        self.messages = [{"role": "system", "content": "You are an Italian tutor whose aim is to help a student learn italian. Only speak italian if you are asked to or if the users prompt is in italian. Otherwise speak english."}]

        # Define the functions that GPT can call
        self.functions = [
            {
                "name": "_pythagrian_theorum",
                "description": "Calculate the hypotenuse of a right triangle given the two legs.",
                "output": "number",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The base of the triangle.",
                        },
                        "y": {"type": "number",
                              "description": "The height of the triangle.",},
                    },
                    "required": ["x", "y"],
                },
            },
            {
                "name": "_create_function",
                "description": "Creates a python function.",
                "output": "string",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the function.",
                        },
                        "description": {
                            "type": "string",
                            "description": "The description of the function.",
                        },
                    },
                    "required": ["name", "description"],
                },
            }
        ]

        self.available_functions = {
                "_pythagrian_theorum": self._pythagrian_theorum,
                "_create_function": self._create_function,
            }  

        # Load Models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Whisper [Ears]
        self.stt_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(self.device)
        self.stt_model.config.forced_decoder_ids = None
        self.stt_processor = WhisperProcessor.from_pretrained("openai/whisper-base")

    def chat_gpt_api(self, user_message):

        '''
        Function to send user message to GPT and get response. It requires an API key from OpenAI and exported to environment variables.

        Parameters
        ----------
        user_message : str
            The message to send to GPT.

        Returns
        -------
        response_message : dict
            The response from GPT.
        '''

        # Send the user message to GPT
        self.messages.append({"role": "user", "content": user_message})
        try:
            response_message = client.chat.completions.create(model=self.model,
            messages=self.messages,
            functions=self.functions)
        except Exception as e:
            print("OpenAI API error:", e)
            return

        # Check if GPT wants to call a function
        if response_message.choices[0].message.function_call:
            # If so, check which function it wants to call

            function_name = response_message.choices[0].message.function_call.name
            # Check for functions that need to run locally
            if function_name=="_create_function":
                print(response_message)
                self._create_function(user_message)
            if function_name=="python":
                print('I want to run the python function')
                print(response_message)
            else:
                self._run_function_api(function_name, response_message)  
        else:
            # If no function call, just send the response message back to the user
            self.messages.append({"role": "assistant", "content": response_message.choices[0].message.content})
            return response_message

    def voice2text(self):
        '''
        Records audio and converts it to text.
        '''

        print("Recording...")
        duration = 5  # seconds
        recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='int16')
        sd.wait()  # Wait for the recording to finish
        print("Done!")

        # Convert to an AudioSegment object
        audio = AudioSegment(
            data=np.array(recording).tobytes(),
            sample_width=recording.dtype.itemsize,
            frame_rate=16000,
            channels=1
        )

        # Resample to 16kHz
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio_data = np.array(audio.get_array_of_samples())
        audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
        audio_data = np.expand_dims(audio_data, axis=0)

        input_features = self.stt_processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(self.device)

        # Prepare attention mask
        attention_mask = (input_features != self.stt_processor.feature_extractor.padding_value).float()

        # Generate with attention mask and language setting
        predicted_ids = self.stt_model.generate(
            input_features,
            attention_mask=attention_mask,
            max_length=100,
            temperature=0.7,
            num_beams=5,
            early_stopping=True,
            forced_decoder_ids=self.stt_processor.get_decoder_prompt_ids(language="en")
            )


        transcription = self.stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]


    def text2voice(self, text):
        '''
        Converts text to speech.
        '''

        mytext = str(text)

        # Language in which you want to convert
        language = "it"

        for i in mytext.split(". "):
            myobj = gTTS(text=i, lang=language, slow=False)
            mp3_fp = BytesIO()
            myobj.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            audio = AudioSegment.from_file(mp3_fp)
            # Speed up the playback
            speed_factor = 1.2  # Change this value as per your requirements
            audio_fast = audio.speedup(playback_speed=speed_factor)

            play(audio_fast,)

    def _pythagrian_theorum(self, x, y):
        return str(np.sqrt(x**2 + y**2))

    def _check_command(self, user_message):


        check_command_conversation = [{"role": "system", "content": "You are a text classifier that determines if a sentence is requesting one of the following commands and output the integer (and only the integer) to the related label. {'Requesting code be run': 0, 'Requesting a pdf be read': 1, 'Requesting code be created': 3, 'Other': 4}"}]
        check_command_conversation.append({"role": "user", "content": f"Classify the following sentence: {user_message}"})

        response = client.chat.completions.create(model=self.model,
        messages=check_command_conversation)


        del check_command_conversation

        return int(response.choices[0].message.content)

    def _create_function(self, user_message):
        print("Creating function...")

    def _run_function_api(self, function_name, response_message):

        try:
            fuction_to_call = self.available_functions[function_name]
        except:
            print("Function not found.")
            return

        function_args = json.loads(response_message.choices[0].message.function_call.arguments)
        function_response = fuction_to_call(
            x=function_args.get("x"),
            y=function_args.get("y"),
        )

        # Step 4: send the info on the function call and function response to GPT
        function_call_message = {
            "role": "assistant",
            "content": None,
            "function_call": response_message.choices[0].message.function_call
        }

        self.messages.append(function_call_message)  
        self.messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  

        second_response = client.chat.completions.create(model=self.model,
        messages=self.messages) 

        self.messages.append({"role": "assistant", "content": second_response.choices[0].message.content}) 

        return second_response

