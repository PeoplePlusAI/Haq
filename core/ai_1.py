from openai import OpenAI
import os
import json
from functools import lru_cache
# portkey
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders, Portkey
# llama index imports 
from llama_index.core import (
    SimpleDirectoryReader, StorageContext, load_index_from_storage, VectorStoreIndex,
    Document, Settings, PromptTemplate
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from utils.bhashini_utils import bhashini_translate
from dotenv import load_dotenv
# from llama_index.legacy.query_engine import FLAREInstructQueryEngine

from utils.bhashini_utils import (
    bhashini_translate
    # bhashini_asr,
    # bhashini_tts
)

# # Templates
# QA_TEMPLATE = PromptTemplate(
#     "Context information is below.\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "Given this information, please answer the question: {query_str}\n"
#     "If you don't know the answer, just say that you don't know. Don't try to make up an answer.\n"
#     "Provide a detailed response and explain your reasoning step by step."
# )

# REFINE_TEMPLATE = PromptTemplate(
#     "The original question is as follows: {query_str}\n"
#     "We have provided an existing answer: {existing_answer}\n"
#     "We have the opportunity to refine the existing answer "
#     "(only if needed) with some more context below.\n"
#     "------------\n"
#     "{context_msg}\n"
#     "------------\n"
#     "Given the new context, refine the original answer to better "
#     "answer the question. If the context isn't useful, return the original answer."
# )

# Load environment variables
load_dotenv(dotenv_path="ops/.env")

# Constants
PERSIST_DIR = "./storage"
DATA_FILE = 'data/Haq_data_v4.txt'
PORTKEY_HEADERS = {
    "x-portkey-api-key": os.getenv("PORTKEY_API_KEY"),
    "x-portkey-provider": "openai",
    "Content-Type": "application/json"
}

# Initialize settings
Settings.chunk_size = 128
Settings.llm = OpenAI(
    model=os.getenv("MODEL_NAME"),
    temperature=0.1,
    api_base=os.getenv("PORTKEY_GATEWAY_URL"),
    default_headers=PORTKEY_HEADERS
)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# openai_api_key = os.getenv("OPENAI_API_KEY")
# port_api_key = os.getenv("PORTKEY_API_KEY")
# model = os.getenv("MODEL_NAME")

@lru_cache(maxsize=1)
def get_index():
    if os.path.exists(PERSIST_DIR):
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))
    
    documents = SimpleDirectoryReader(input_files=[DATA_FILE]).load_data()
    document = Document(text="\n\n".join(doc.text for doc in documents))
    index = VectorStoreIndex.from_documents([document])
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

@lru_cache(maxsize=100)
def llama_index_rag(input_message):
    query_engine = get_index().as_query_engine(similarity_top_k=2)
    
    response = query_engine.query(input_message)
    return str(response)

def ragindex(chat_id, input_message):
    return llama_index_rag(input_message)


def bhashini_text_chat(chat_id, text, lang):
    input_message = bhashini_translate(text, lang, "en")
    response_en = ragindex(chat_id, input_message)
    response = bhashini_translate(response_en, "en", lang)
    return response, response_en
