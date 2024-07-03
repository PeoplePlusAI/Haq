from openai import OpenAI
import os
import redis
import json

# portkey
# from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders
# from portkey_ai.llms.llama_index import PortkeyLLM

#from llama_index.llms.portkey import Portkey
# from llama_index.core.llms import ChatMessage
# import portkey as pk
from llama_index.core import Settings

# import openai files
from utils.openai_utils import (
    create_thread,
    upload_message,
    get_run_status,
    get_assistant_message,
    create_assistant,
    transcribe_audio,
) # generate_audio

# llama index imports 
# from llama_index.legacy.text_splitter import SentenceSplitter
from llama_index.legacy import (
    SimpleDirectoryReader, StorageContext,load_index_from_storage #LLMPredictor, ServiceContext, KeywordTableIndex,
)

from llama_index.core import (
    SimpleDirectoryReader, StorageContext, load_index_from_storage, VectorStoreIndex
)
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from utils.redis_utils import (
    get_redis_value,
    set_redis,
)

from dotenv import load_dotenv
import sqlite3
import os
load_dotenv(
    dotenv_path="ops/.env",
)

openai_api_key = os.getenv("OPENAI_API_KEY")
# port_api_key = os.getenv("PORTKEY_API_KEY")
# port_virtual_key = os.getenv("PORTKEY_VIRTUAL_KEY")

llm = OpenAI(model="gpt-4-turbo", temperature=0.1)
client = OpenAI(
    api_key=openai_api_key,
)

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url="https://api.portkey.ai/v1", ## Point to Portkey's gateway URL
#     default_headers= {
#         "x-portkey-api-key": "GAR9058m6pQDVDohpyDlhz98iv4=",
#         "x-portkey-provider": "openai",
#         "Content-Type": "application/json"
#     }
# )

def store_message(input_message, response_message):
    # Store to disk
    with open('ops/response.txt', 'a') as file:
        file.write(input_message + '\n')

    # Store to database
    if not os.path.exists('ops/database.db'):
        open('ops/database.db', 'w').close()

    # Connect to the database
    conn = sqlite3.connect('ops/database.db')

    # Create a cursor object
    cursor = conn.cursor()

    # Create a table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_content TEXT,
            response_content TEXT
        )
    ''')

    # Insert the message into the table
    cursor.execute('INSERT INTO messages (input_content, response_content) VALUES (?, ?)', (input_message, response_message))

    # Commit the changes
    conn.commit()

    # Close the connection
    conn.close()

def llama_index_rag(input_message):
    
    documents = SimpleDirectoryReader(input_files=['HD_app_Commonerrors.txt']).load_data()
    print(type(documents))
    document = Document(text="\n\n".join([doc.text for doc in documents]))
    
    # port key config
    # headers = createHeaders(api_key=port_api_key, mode="openai")
    # headers= {
    #     "x-portkey-api-key": "GAR9058m6pQDVDohpyDlhz98iv4=",
    #     "x-portkey-provider": "openai",
    #     "Content-Type": "application/json"
    # }
    #llm = OpenAI(model="gpt-4-0125-preview", temperature=0.1)
    
    service_context = ServiceContext.from_defaults(llm=llm)
    # llm=OpenAI(model="gpt-4", temperature=0)
    # portkey = PortkeyLLM(api_key="PORTKEY_API_KEY", virtual_key="VIRTUAL_KEY")
    #Settings.llm = llm

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
        #index = KeywordTableIndex.from_documents(documents, service_context=service_context)
    
    query_engine = index.as_query_engine(similarity_top_k=2, llm=llm)
    
    flare_query_engine = FLAREInstructQueryEngine(
        query_engine=query_engine,
        service_context=service_context,
        max_iterations=3,
        verbose=True,
        )
    print(type(flare_query_engine))
    
    response = flare_query_engine.query(f"{input_message}")
    print(str(response))
    
    # store the input message
    # store_message(input_message)
    store_message(input_message, str(response))
    
    return str(response)

def ragindex(chat_id, input_message):
    res = llama_index_rag(input_message)
    print(res)
    print(type(res))
    # json.dumps(res)
    return res
  
def audio_chat(chat_id, audio_file):
    input_message = transcribe_audio(audio_file, client)
    print(f"The input message is : {input_message}")
    assistant_message, history =  chat(chat_id, input_message)
    response_audio = generate_audio(assistant_message, client)
    return response_audio, history

def chat(chat_id, input_message):
    try:
        assistant_id = get_redis_value("assistant_id")
        print(assistant_id)
    except Exception as e:
        print(e)
    
    history = get_redis_value(chat_id)
    if history == None:
        history = {
            "thread_id": None,
            "run_id": None,
            "status": None,
        }
    else:
        history = json.loads(history)
    thread_id = history.get("thread_id")
    run_id = history.get("run_id")
    status = history.get("status")

    try:
        run = client.beta.threads.runs.retrieve(thread_id, run_id)
    except Exception as e:
        run = None
    try:
        thread = client.beta.threads.retrieve(thread_id)
    except Exception as e:
        thread = create_thread(client)

    if status == "completed" or status == None:
        
        run = upload_message(client, thread.id, input_message, assistant.id)
        run, status = get_run_status(run, client, thread)

        assistant_message = get_assistant_message(client, thread.id)

        history = {
            "thread_id": thread.id,
            "run_id": run.id,
            "status": status,
        }
        history = json.dumps(history)
        set_redis(chat_id, history)
    
    if status == "requires_action":
        if run:
            tools_to_call = run.required_action.submit_tool_outputs.tool_calls
        else:
            run, status = get_run_status(run, client, thread)
            tools_to_call = run.required_action.submit_tool_outputs.tool_calls

        for tool in tools_to_call:
            func_name = tool.function.name
            print(f"Function name: {func_name}")
            parameters = json.loads(tool.function.arguments)
            # parameters["auth_token"] = auth_token
            # parameters["username"] = username
            print(f"Parameters: {parameters}")

            tool_output_array = []

            if func_name == "rag":
                answer = rag(parameters)
                if answer:
                    tool_output_array.append(
                        {
                            "tool_call_id": tool.id,
                            "output": answer
                        }
                    )
                    run = client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=tool_output_array
                    )
                    run, status = get_run_status(run, client, thread)

                    message = get_assistant_message(client, thread.id)

                    history = {
                        "thread_id": thread.id,
                        "run_id": run.id,
                        "status": status,
                    }
                    history = json.dumps(history)
                    set_redis(chat_id, history)
                    return message, history
                else:
                    return "Complaint failed", history
                
            elif func_name == "search_complaint":
                complaint = search_complaint(parameters)
                if complaint:
                    tool_output_array.append(
                        {
                            "tool_call_id": tool.id,
                            "output": complaint["ServiceWrappers"][0]["service"]["applicationStatus"]
                        }
                    )
                    run = client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=tool_output_array
                    )
                    run, status = get_run_status(run, client, thread)

                    message = get_assistant_message(client, thread.id)

                    history = {
                        "thread_id": thread.id,
                        "run_id": run.id,
                        "status": status,
                    }
                    history = json.dumps(history)
                    set_redis(chat_id, history)
                    return message, history
                else:
                    return "Message not found", history
                
    return assistant_message, history
