from openai import OpenAI
import os
import redis
import json

# portkey
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders
#from portkey_ai.llms.llama_index import PortkeyLLM

from utils.bhashini_utils import (
    bhashini_translate
    # bhashini_asr,
    # bhashini_tts
)

# import openai files
# from utils.openai_utils import (
#     create_thread,
#     upload_message,
#     get_run_status,
#     get_assistant_message,
#     create_assistant,
#     transcribe_audio,
# ) # generate_audio

# llama index imports 
# from llama_index.legacy.text_splitter import SentenceSplitter
from llama_index.legacy import (
    SimpleDirectoryReader, StorageContext,load_index_from_storage
)
from llama_index.legacy import Document
from llama_index.legacy import VectorStoreIndex
from llama_index.legacy import ServiceContext
from llama_index.legacy.llms import OpenAI
from llama_index.legacy.embeddings import OpenAIEmbedding
from llama_index.legacy.query_engine import FLAREInstructQueryEngine


from dotenv import load_dotenv
load_dotenv(
    dotenv_path="ops/.env",
)

openai_api_key = os.getenv("OPENAI_API_KEY")
port_api_key = os.getenv("PORTKEY_API_KEY")
model = os.getenv("MODEL_NAME")
#llm = ChatOpenAI(model_name="gpt-4", temperature=0)
# client = OpenAI(
#     api_key=openai_api_key,
# )

def llama_index_rag(input_message):
    documents = SimpleDirectoryReader(input_files=['HD_app_Commonerrors.txt']).load_data()
    print(type(documents))
    document = Document(text="\n\n".join([doc.text for doc in documents]))
    
    # port key config
    # headers = createHeaders(api_key=port_api_key, mode="openai")
    headers= {
        "x-portkey-api-key": "GAR9058m6pQDVDohpyDlhz98iv4=",
        "x-portkey-provider": "openai",
        "Content-Type": "application/json"
    }
    try:    
        llm = OpenAI(model=model, temperature=0.1, api_base=PORTKEY_GATEWAY_URL, default_headers=headers)
        # else use gpt-4
    except Exception as e:
        print(e)
        llm = OpenAI(model=model, temperature=0.1)
    
    service_context = ServiceContext.from_defaults(llm=llm)
    # llm=OpenAI(model="gpt-4", temperature=0)
    # service_context = ServiceContext.from_defaults(llm=llm)
    # portkey = PortkeyLLM(api_key="PORTKEY_API_KEY", virtual_key="VIRTUAL_KEY")
    # service_context = ServiceContext.from_defaults(llm=portkey)

    #service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
    PERSIST_DIR = "./storage"
    
    if not os.path.exists(PERSIST_DIR):
            index = VectorStoreIndex.from_documents([document],service_context=service_context)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
    # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    
    query_engine = index.as_query_engine(similarity_top_k=2)
    
    flare_query_engine = FLAREInstructQueryEngine(
        query_engine=query_engine,
        service_context=service_context,
        max_iterations=1, # hyper-parameter - can vary from 1 to 7
        verbose=False, # hyper-parameter - can vary from True to False
        )
    print(type(flare_query_engine))
    
    response = flare_query_engine.query(f"{input_message}")
    print(str(response))
    return str(response)

def ragindex(chat_id, input_message):
    res = llama_index_rag(input_message)
    print(res)
    print(type(res))
    # json.dumps(res)
    return res

def bhashini_text_chat(chat_id, text, lang): 
    """
    bhashini text chat logic
    """
    input_message = bhashini_translate(text, lang, "en")
    response_en = ragindex(chat_id, input_message)
    response = bhashini_translate(response_en, "en", lang)
    return response, response_en
# # def audio_chat(chat_id, audio_file):
# #     input_message = transcribe_audio(audio_file, client)
# #     print(f"The input message is : {input_message}")
# #     assistant_message, history =  chat(chat_id, input_message)
# #     response_audio = generate_audio(assistant_message, client)
# #     return response_audio, history

# def chat(chat_id, input_message):
#     try:
#         assistant_id = get_redis_value("assistant_id")
#         print(assistant_id)
#     except Exception as e:
#         print(e)
    
#     history = get_redis_value(chat_id)
#     if history == None:
#         history = {
#             "thread_id": None,
#             "run_id": None,
#             "status": None,
#         }
#     else:
#         history = json.loads(history)
#     thread_id = history.get("thread_id")
#     run_id = history.get("run_id")
#     status = history.get("status")

#     try:
#         run = client.beta.threads.runs.retrieve(thread_id, run_id)
#     except Exception as e:
#         run = None
#     try:
#         thread = client.beta.threads.retrieve(thread_id)
#     except Exception as e:
#         thread = create_thread(client)

#     if status == "completed" or status == None:
        
#         run = upload_message(client, thread.id, input_message, assistant.id)
#         run, status = get_run_status(run, client, thread)

#         assistant_message = get_assistant_message(client, thread.id)

#         history = {
#             "thread_id": thread.id,
#             "run_id": run.id,
#             "status": status,
#         }
#         history = json.dumps(history)
#         set_redis(chat_id, history)
    
#     if status == "requires_action":
#         if run:
#             tools_to_call = run.required_action.submit_tool_outputs.tool_calls
#         else:
#             run, status = get_run_status(run, client, thread)
#             tools_to_call = run.required_action.submit_tool_outputs.tool_calls

#         for tool in tools_to_call:
#             func_name = tool.function.name
#             print(f"Function name: {func_name}")
#             parameters = json.loads(tool.function.arguments)
#             # parameters["auth_token"] = auth_token
#             # parameters["username"] = username
#             print(f"Parameters: {parameters}")

#             tool_output_array = []

#             if func_name == "rag":
#                 answer = rag(parameters)
#                 if answer:
#                     tool_output_array.append(
#                         {
#                             "tool_call_id": tool.id,
#                             "output": answer
#                         }
#                     )
#                     run = client.beta.threads.runs.submit_tool_outputs(
#                         thread_id=thread.id,
#                         run_id=run.id,
#                         tool_outputs=tool_output_array
#                     )
#                     run, status = get_run_status(run, client, thread)

#                     message = get_assistant_message(client, thread.id)

#                     history = {
#                         "thread_id": thread.id,
#                         "run_id": run.id,
#                         "status": status,
#                     }
#                     history = json.dumps(history)
#                     set_redis(chat_id, history)
#                     return message, history
#                 else:
#                     return "Complaint failed", history
                
#             elif func_name == "search_complaint":
#                 complaint = search_complaint(parameters)
#                 if complaint:
#                     tool_output_array.append(
#                         {
#                             "tool_call_id": tool.id,
#                             "output": complaint["ServiceWrappers"][0]["service"]["applicationStatus"]
#                         }
#                     )
#                     run = client.beta.threads.runs.submit_tool_outputs(
#                         thread_id=thread.id,
#                         run_id=run.id,
#                         tool_outputs=tool_output_array
#                     )
#                     run, status = get_run_status(run, client, thread)

#                     message = get_assistant_message(client, thread.id)

#                     history = {
#                         "thread_id": thread.id,
#                         "run_id": run.id,
#                         "status": status,
#                     }
#                     history = json.dumps(history)
#                     set_redis(chat_id, history)
#                     return message, history
#                 else:
#                     return "Message not found", history
                
#     return assistant_message, history
