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

# pip install -U sentence-transformers
# from sentence_transformers import SentenceTransformer

from utils.bhashini_utils import bhashini_translate # bhashini_asr,# bhashini_tts)

import logging
from typing import Optional
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.callbacks import CallbackManager, LlamaDebugHandler

# from llama_index.response.pprint_utils import pprint_response
# pprint_response(response, show_source=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain import hub
# from llama_index.core.prompts import LangchainPromptTemplate
from llama_index.core import PromptTemplate

# TO ARTIFICALLY GENERATE Q&A
from llama_index.core.evaluation import generate_question_context_pairs
# Prompt to generate questions
qa_generate_prompt_tmpl = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. The questions should not contain options, not start with Q1/ Q2. \
Restrict the questions to the context information provided.\
"""

class CustomQueryEngine(RetrieverQueryEngine):
    def __init__(self, custom_str, refine_str):
        self.custom_str = custom_str
        self.refine_str = refine_str
    
    def custom_query(self, query_str: str):
        # Retrieve nodes relevant to the query
        nodes = self.retriever.retrieve(query_str)
        
        qa_dataset = generate_question_context_pairs(
            nodes, llm=llm, num_questions_per_chunk=2, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
        )
        # The returned result is a EmbeddingQAFinetuneDataset object (containing queries, relevant_docs, and corpus).
        
        # Generate the context string
        context_str = "\n\n".join([n.node.get_context() for n in nodes])
        
        # print or log the context string
        print("Context string:", context_str)
        
        # Call the superclass's query method to generate a response
        return super().query(query_str)

# Templates
QA_TEMPLATE = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
    "If you don't know the answer, just say that you don't know. Don't try to make up an answer.\n"
    "Provide a detailed response and explain your reasoning step by step."
)

REFINE_TEMPLATE = PromptTemplate(
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. If the context isn't useful, return the original answer."
)

# Load environment variables
load_dotenv(dotenv_path="ops/.env")
# llm = OpenAI()
# Constants
PERSIST_DIR = "./storage"
DATA_FILE = 'data/Haq_data_v4.txt'
PORTKEY_HEADERS = {
    "x-portkey-api-key": os.getenv("PORTKEY_API_KEY"),
    "x-portkey-provider": "openai",
    "Content-Type": "application/json"
}
# Initialize settings
Settings.chunk_size = 512
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
def get_or_create_index():
    if os.path.exists(PERSIST_DIR):
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))
    
    documents = SimpleDirectoryReader(input_files=[DATA_FILE]).load_data()
    document = Document(text="\n\n".join(doc.text for doc in documents))
    index = VectorStoreIndex.from_documents([document])
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

SIMILARITY_CUTOFF = 0.7
TOP_K = 3

# RETREIVER PART
from llama_index.core.evaluation import RetrieverEvaluator

def create_custom_query_engine(index: VectorStoreIndex) -> RetrieverQueryEngine:
    """Create a custom query engine with advanced retrieval and postprocessing."""
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], 
        retriever=retriever)

    # retriever_evaluator.evaluate(
    #     query="query", expected_ids=["node_id1", "node_id2"]
    # )
    # eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
    
    return RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)],
        text_qa_template=QA_TEMPLATE,
        refine_template=REFINE_TEMPLATE,
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)],
        text_qa_template=lc_prompt_tmpl,
    )
    return query_engine

def semantic_cache(func):
    """Simple semantic caching decorator."""
    cache = {}
    
    def wrapper(*args, **kwargs):
        query = args[0] if args else kwargs.get('input_message', '')
        for cached_query, cached_response in cache.items():
            if semantic_similarity(query, cached_query) > 0.9:  # Adjust threshold as needed
                logger.info("Using cached response for similar query.")
                return cached_response
        result = func(*args, **kwargs)
        cache[query] = result
        return result
    
    return wrapper

def semantic_similarity(query1: str, query2: str) -> float:
    """Compute semantic similarity between two queries."""
    # Implement semantic similarity calculation here
    # This is a placeholder and should be replaced with actual implementation
    return 0.5  # Placeholder value


# @lru_cache(maxsize=100)
@semantic_cache
def llama_index_rag(input_message):
    # query_engine = get_or_create_index().as_query_engine(similarity_top_k=2)
    query_engine = create_custom_query_engine(get_or_create_index())
    # debug_handler = LlamaDebugHandler(print_trace_on_end=True)
    # callback_manager = CallbackManager([debug_handler])
    try:
        response = query_engine.query(input_message)
        logger.info(f"Query: {input_message}")
        logger.info(f"Response: {response}")
        return str(response)
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        return "An error occurred while processing your query. Please try again."

    # response = query_engine.query(input_message)
    # return str(response)

    # for streaming
    # stream = llm.stream_complete(input_message)
    # for r in stream:
    #     print(r.delta, end="", flush = True)

def ragindex(chat_id: str, input_message: str) -> str:
    """Wrapper function to call llama_index_rag."""
    res = llama_index_rag(input_message)
    logger.info(f"Chat ID: {chat_id}, Query: {input_message}")
    logger.info(f"Response type: {type(res)}, Response: {res}")
    return res


def bhashini_text_chat(chat_id, text, lang):
    input_message = bhashini_translate(text, lang, "en")
    response_en = ragindex(chat_id, input_message)
    response = bhashini_translate(response_en, "en", lang)
    return response, response_enfrom openai import OpenAI
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

# pip install -U sentence-transformers
# from sentence_transformers import SentenceTransformer

from utils.bhashini_utils import bhashini_translate # bhashini_asr,# bhashini_tts)

import logging
from typing import Optional
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.callbacks import CallbackManager, LlamaDebugHandler

# from llama_index.response.pprint_utils import pprint_response
# pprint_response(response, show_source=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain import hub
# from llama_index.core.prompts import LangchainPromptTemplate
from llama_index.core import PromptTemplate

# TO ARTIFICALLY GENERATE Q&A
from llama_index.core.evaluation import generate_question_context_pairs
# Prompt to generate questions
qa_generate_prompt_tmpl = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. The questions should not contain options, not start with Q1/ Q2. \
Restrict the questions to the context information provided.\
"""

class CustomQueryEngine(RetrieverQueryEngine):
    def __init__(self, custom_str, refine_str):
        self.custom_str = custom_str
        self.refine_str = refine_str
    
    def custom_query(self, query_str: str):
        # Retrieve nodes relevant to the query
        nodes = self.retriever.retrieve(query_str)
        
        qa_dataset = generate_question_context_pairs(
            nodes, llm=llm, num_questions_per_chunk=2, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
        )
        # The returned result is a EmbeddingQAFinetuneDataset object (containing queries, relevant_docs, and corpus).
        
        # Generate the context string
        context_str = "\n\n".join([n.node.get_context() for n in nodes])
        
        # print or log the context string
        print("Context string:", context_str)
        
        # Call the superclass's query method to generate a response
        return super().query(query_str)

# Templates
QA_TEMPLATE = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
    "If you don't know the answer, just say that you don't know. Don't try to make up an answer.\n"
    "Provide a detailed response and explain your reasoning step by step."
)

REFINE_TEMPLATE = PromptTemplate(
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. If the context isn't useful, return the original answer."
)

# Load environment variables
load_dotenv(dotenv_path="ops/.env")
# llm = OpenAI()
# Constants
PERSIST_DIR = "./storage"
DATA_FILE = 'data/Haq_data_v4.txt'
PORTKEY_HEADERS = {
    "x-portkey-api-key": os.getenv("PORTKEY_API_KEY"),
    "x-portkey-provider": "openai",
    "Content-Type": "application/json"
}
# Initialize settings
Settings.chunk_size = 512
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
def get_or_create_index():
    if os.path.exists(PERSIST_DIR):
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))
    
    documents = SimpleDirectoryReader(input_files=[DATA_FILE]).load_data()
    document = Document(text="\n\n".join(doc.text for doc in documents))
    index = VectorStoreIndex.from_documents([document])
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

SIMILARITY_CUTOFF = 0.7
TOP_K = 3

# RETREIVER PART
from llama_index.core.evaluation import RetrieverEvaluator

def create_custom_query_engine(index: VectorStoreIndex) -> RetrieverQueryEngine:
    """Create a custom query engine with advanced retrieval and postprocessing."""
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], 
        retriever=retriever)

    # retriever_evaluator.evaluate(
    #     query="query", expected_ids=["node_id1", "node_id2"]
    # )
    # eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
    
    return RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)],
        text_qa_template=QA_TEMPLATE,
        refine_template=REFINE_TEMPLATE,
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)],
        text_qa_template=lc_prompt_tmpl,
    )
    return query_engine

def semantic_cache(func):
    """Simple semantic caching decorator."""
    cache = {}
    
    def wrapper(*args, **kwargs):
        query = args[0] if args else kwargs.get('input_message', '')
        for cached_query, cached_response in cache.items():
            if semantic_similarity(query, cached_query) > 0.9:  # Adjust threshold as needed
                logger.info("Using cached response for similar query.")
                return cached_response
        result = func(*args, **kwargs)
        cache[query] = result
        return result
    
    return wrapper

def semantic_similarity(query1: str, query2: str) -> float:
    """Compute semantic similarity between two queries."""
    # Implement semantic similarity calculation here
    # This is a placeholder and should be replaced with actual implementation
    return 0.5  # Placeholder value


# @lru_cache(maxsize=100)
@semantic_cache
def llama_index_rag(input_message):
    # query_engine = get_or_create_index().as_query_engine(similarity_top_k=2)
    query_engine = create_custom_query_engine(get_or_create_index())
    # debug_handler = LlamaDebugHandler(print_trace_on_end=True)
    # callback_manager = CallbackManager([debug_handler])
    try:
        response = query_engine.query(input_message)
        logger.info(f"Query: {input_message}")
        logger.info(f"Response: {response}")
        return str(response)
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        return "An error occurred while processing your query. Please try again."

    # response = query_engine.query(input_message)
    # return str(response)

    # for streaming
    # stream = llm.stream_complete(input_message)
    # for r in stream:
    #     print(r.delta, end="", flush = True)

def ragindex(chat_id: str, input_message: str) -> str:
    """Wrapper function to call llama_index_rag."""
    res = llama_index_rag(input_message)
    logger.info(f"Chat ID: {chat_id}, Query: {input_message}")
    logger.info(f"Response type: {type(res)}, Response: {res}")
    return res


def bhashini_text_chat(chat_id, text, lang):
    input_message = bhashini_translate(text, lang, "en")
    response_en = ragindex(chat_id, input_message)
    response = bhashini_translate(response_en, "en", lang)
    return response, response_en