from dotenv import load_dotenv
from utils.redis_utils import set_redis
import time
import os
import random
from pydub import AudioSegment

from utils.bhashini_utils import (
    bhashini_translate,
    bhashini_asr,
    bhashini_tts
)

load_dotenv(
    dotenv_path="ops/.env",
)
'''
with open("prompts/main.txt", "r") as file:
    main_prompt = file.read()

print(main_prompt)
'''

openai_api_key = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("ASSISTANT_ID")
model_name = os.getenv("MODEL_NAME")
#OPENAI FUNCTION CALLS

# def create_assistant(client, assistant_id):
#     try:
#         assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
#         return assistant
#     except Exception as e:
#         assistant = client.beta.assistants.create(
#         name="Complaint Assistant",
#         instructions="You ara a helpful complaint assistant who will collect information about a complaint and raise the complaint. You are talking to common citizens who are not tech savvy, so ask questions one by one. You will also have to search for complaints raised by the user.",
#         model="gpt-4",
#         tools=[
#                 #{
#                 #    "type": "function",
#                 #    "function": authenticate_user
#                 #},
#                 {
#                     "type": "function",
#                     "function": raise_complaint
#                 },
#                 {
#                     "type": "function",
#                     "function": search_complaint
#                 }
#             ]
#         )
#         set_redis("assistant_id", assistant.id)
#         return assistant

# def create_thread(client):
#     thread = client.beta.threads.create()
#     return thread

# def upload_message(client, thread_id, input_message, assistant_id):
#     message = client.beta.threads.messages.create(
#         thread_id=thread_id,
#         role="user",
#         content=input_message
#     )

#     run = client.beta.threads.runs.create(
#         thread_id=thread_id,
#         assistant_id=assistant_id,
#     )
    
#     return run

# def get_run_status(run, client, thread):
#     i = 0

#     while run.status not in ["completed", "failed", "requires_action"]:
#         if i>0:
#             time.sleep(10)

#         run = client.beta.threads.runs.retrieve(
#             thread_id=thread.id,
#             run_id=run.id,
#         )
#         i += 1
#     return run, run.status

# def get_assistant_message(client, thread_id):
#     messages = client.beta.threads.messages.list(
#         thread_id=thread_id,
#     )
#     return messages.data[0].content[0].text.value


def transcribe_audio(audio_file, client):
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )
    return transcript.text

def generate_audio(text, client):
    response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
    return response


def get_duration_pydub(file_path):
    try:
        audio_file = AudioSegment.from_file(file_path)
        duration = audio_file.duration_seconds
        return duration
    except Exception as e:
        print(f"Error occurred while getting duration: {e}")
        return None

def get_random_wait_messages(not_always=False, lang="en"):
    messages = [
        "Please wait",
        "I am processing your request",
        "Hold on",
        "I am on it",
        "I am working on it",
    ]
    if not_always:
        rand = random.randint(0, 2)
        if rand == 1:
            random_message = random.choice(messages)
            random_message = bhashini_translate(random_message, "en", lang)
        else:
            random_message = ""
    else:
        random_message = random.choice(messages)
        random_message = bhashini_translate(random_message, "en", lang)
    return random_message

# def clean(query):

#   # regex to clean the input string
#   # Remove non-alphanumeric characters except spaces and periods
#   clean_query = re.sub(r"[^a-zA-Z0-9\s.]", "", query)


#   # Check whether the prompt is complete - validation layer
#   # Basic check for length and content
#   if len(clean_query) < 5 or clean_query.strip(".") == "":
#       return "Try Again. Please provide more details about your complaint."

#   # reprompt using LLM itself
#   # If input passes basic validation but still needs clarification

#   instruction = f"""
#   Rewrite the user's input for filing a complaint so that it is clear and specific. If it is incomplete or blank, output 'Try Again'
#   """

#   messages = [{"role": "system", "content": instruction},
#               # {"role": "assistant", "content": "What is your complaint? Also mention the locality from where the complaint is being filed."},
#               {"role": "user", "content": clean_query}
#             ]
#   response = client.chat.completions.create(
#       model="gpt-4-1106-preview",
#       messages=messages,
#   )
#   response_message = response.choices[0].message

#   # Check if LLM response is asking for a retry
#   if "Try Again" in response_message:
#       return "Try Again. Please provide a clearer description of your complaint."

#   return response_message
