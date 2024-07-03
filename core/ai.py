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
from llama_index.core import (
    SimpleDirectoryReader, StorageContext, load_index_from_storage, VectorStoreIndex
)
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
# from llama_index.legacy.query_engine import FLAREInstructQueryEngine

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
    documents = SimpleDirectoryReader(input_files=['HD_data_v1.txt']).load_data()
    # to use scheme information alongside
    # documents = SimpleDirectoryReader(input_files=['data/HD_data_v2.txt']).load_data()
    
    # print(type(documents))
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
    
    Settings.llm = llm
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.embed_model = embed_model
    Settings.chunk_size = 128 # 512
    
    # service_context = ServiceContext.from_defaults(llm=llm) # or llm = portkey
    # service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
    # portkey = PortkeyLLM(api_key="PORTKEY_API_KEY", virtual_key="VIRTUAL_KEY")
    
    PERSIST_DIR = "./storage"
    
    if not os.path.exists(PERSIST_DIR):
            index = VectorStoreIndex.from_documents([document]) # service_context=service_context
            index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
    # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    
    query_engine = index.as_query_engine(similarity_top_k=2)
    
    # flare_query_engine = FLAREInstructQueryEngine(
    #     query_engine=query_engine,
    #     service_context=service_context,
    #     max_iterations=1,
    #     verbose=False,
    #     )
    # print(type(flare_query_engine))
    
    response = query_engine.query(f"{input_message}")
    
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
