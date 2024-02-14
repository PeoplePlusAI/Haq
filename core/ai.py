from openai import OpenAI
import os
import redis

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
from llama_index.text_splitter import SentenceSplitter
from llama_index import SimpleDirectoryReader
from llama_index import Document
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.query_engine import FLAREInstructQueryEngine

# importing Langchain modules
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
#from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
#from langchain_community.document_loaders import WebBaseLoaderclear
# from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain_community.vectorstores import Chroma # FAISS or Weaviate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.document_loaders import PyPDFLoader, DirectoryLoader

from utils.redis_utils import (
    get_redis_value,
    set_redis,
)
import json
import time
import os

from dotenv import load_dotenv

load_dotenv(
    dotenv_path="ops/.env",
)

try:
    assistant_id = get_redis_value("assistant_id")
    print(assistant_id)
except Exception as e:
    print(e)

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

client = OpenAI(
    api_key=openai_api_key,
)
# assistant_id = assistant.id
#assistant = create_assistant(client, assistant_id)

def langchain_splitter(text):

    # print(pages, "\n", len(pages))
    # sep_list = ["\n"," "] # try tune this hyperparameter . like use any regex function too "(?<=\. )".
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,
                                                    chunk_overlap=20,
                                                    #length_function=len,
                                                    #is_separator_regex=False
                                                    #  add_start_index=True)
    )
    texts = text_splitter.create_documents([text])
    # or use pages directly
    return texts

def llama_index_splitter(): #text
    
    splitter = SentenceSplitter(chunk_size=200,chunk_overlap=15)
    # Load up your document
    documents = SimpleDirectoryReader(input_files=['HD_app_Commonerrors.txt']).load_data()
    # documents = SimpleDirectoryReader(input_files=["data.txt"]).load_data()
    nodes = splitter.get_nodes_from_documents(documents)
    #print(documents[0], "/n", len(documents))
    #print(nodes)
    print(type(documents))
    return documents # or nodes
    
def llama_index_rag(documents, input_message):
    document = Document(text="\n\n".join([doc.text for doc in documents]))
    
    llm=OpenAI(model="gpt-4", temperature=0)
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)
    
    # llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    #embed_model = OpenAIEmbedding()
    #service_context = ServiceContext.from_defaults(llm=llm)
    #service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_documents([document],service_context=service_context)
    
    query_engine = index.as_query_engine(similarity_top_k=2)
    
    flare_query_engine = FLAREInstructQueryEngine(
        query_engine=query_engine,
        service_context=service_context,
        max_iterations=7,
        verbose=True,
        )
    print(type(flare_query_engine))
    
    response = flare_query_engine.query(f"{input_message}")
    print(str(response))
    return str(response)

    
def pdf_splitter(input_file):
    # from unstructured.partition.pdf import partition_pdf
    # from unstructured.staging.base import elements_to_json
    
    # filename = "static/SalesforceFinancial.pdf"

    # Extracts the elements from the PDF
    elements = partition_pdf(
        filename=filename,
        # Unstructured Helpers
        strategy="hi_res", 
        infer_table_structure=True, 
        model_name="yolox"
    )

def retriever(input_message, vectorstore):
    # retriever = vectorstore.as_retriever()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}) # can use k = 6
    retrieved_docs = retriever.get_relevant_documents(f"{input_message}")
    # retrieved_docs = retriever.invoke(f"{input_message}")
    print(retrieved_docs[0].page_content)
    
    return retrieved_docs
    
    '''Method 2 :
    prompt = hub.pull("rlm/rag-prompt")
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    docs = vectorstore.similarity_search(input_message,k=3)
    
    rag_chain = (
        {"context": retriever | format_docs(docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    return rag_chain.invoke(f"{input_message}")
    '''
def ragindex(chat_id, input_message):
    docu = llama_index_splitter()
    res = llama_index_rag(docu, input_message)
    print(res)
    print(type(res))
    # json.dumps(res)
    return res

def raglang(chat_id, input_message):
    # loader = TextLoader("Svanidhi.txt") 
    loader = TextLoader('HD_app_Commonerrors.txt')
    #loader = TextLoader('data.txt') # any text or pdf document
    pages = loader.load()
    # Split text into chunks
    docs = langchain_splitter(pages)
    # llama_index_splitter(pages)
    
    # vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())
    persist_directory = '/'
    vectorstore = Chroma.from_documents(documents=docs,
                                 embedding=OpenAIEmbeddings(),
                                 persist_directory=persist_directory)
    vectorstore.persist()
    
    query = f"{input_message}"
    
    try:
        # check
        print(vectorstore._collection.count())
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(),
            chain_type="refine" # NOTE TO SELF: TRY chain_type = "map_reduce" or "map_rerank" IN NEXT ITERATION
        )

        # Build prompt
        template_string = """You help answer queries relted to common problems related to Indian government schemes.
        Question: {query}
        Helpful Answer:"""
        # Build prompt
        prompt_template = ChatPromptTemplate.from_template(template_string)

        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=prompt_template)
        # QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # Run chain
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        result = qa_chain({"query": query})
        print(result["result"]) # shows the final result
        # print(result["source_documents"][0]) # shows the source document too
    except Exception as e:
        print(e)
    
    '''
    docs = vectorstore.similarity_search(
        query,
        k=3,
        filter={"source":"data.txt"}
    )
    '''
    
    # Index Method
    '''
    from langchain.indexes import VectorstoreIndexCreator
    index = VectorstoreIndexCreator().from_loaders([loader])
    
    query = f"{input_message}"
    return index.query_with_sources(query)['answer']
    '''
    
    # alternate way:
    '''
        except Exception as e:
            print("2")
            loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                    )
                ),
            )
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings())
        
        doc = retriever(input_message, vectorstore)
        return doc
    '''

    '''Method 2 :
    question = "What are the approaches to Task Decomposition?"
    docs = vectorstore.similarity_search(question)
    print(docs[0])
    '''

#assistant = create_assistant(client, assistant_id)
    
def audio_chat(chat_id, audio_file):
    input_message = transcribe_audio(audio_file, client)
    print(f"The input message is : {input_message}")
    assistant_message, history =  chat(chat_id, input_message)
    response_audio = generate_audio(assistant_message, client)
    return response_audio, history

def chat(chat_id, input_message):
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
            # username = USERNAME
            # auth_token = get_auth_token(
            #     {
            #         "username": username,
            #         "password": PASSWORD
            #     }
            # )
            func_name = tool.function.name
            print(f"Function name: {func_name}")
            parameters = json.loads(tool.function.arguments)
            # parameters["auth_token"] = auth_token
            # parameters["username"] = username
            print(f"Parameters: {parameters}")

            tool_output_array = []
            """
            if func_name == "authenticate_user":
                auth_token = get_auth_token(parameters)
                if auth_token:
                    tool_output_array.append(
                        {
                            "tool_call_id": tool.id,
                            "output": auth_token
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
                    return "Authentication failed", history
                """

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
                    return "Complaint not found", history
                
    return assistant_message, history

# Advanced RAG using chink size optimisation
'''
from llama_index import ServiceContext
from llama_index.param_tuner.base import ParamTuner, RunResult
from llama_index.evaluation import SemanticSimilarityEvaluator, BatchEvalRunner

### Recipe
### Perform hyperparameter tuning as in traditional ML via grid-search
### 1. Define an objective function that ranks different parameter combos
### 2. Build ParamTuner object
### 3. Execute hyperparameter tuning with ParamTuner.tune()

# 1. Define objective function
def objective_function(params_dict):
    chunk_size = params_dict["chunk_size"]
    docs = params_dict["docs"]
    top_k = params_dict["top_k"]
    eval_qs = params_dict["eval_qs"]
    ref_response_strs = params_dict["ref_response_strs"]

    # build RAG pipeline
    index = _build_index(chunk_size, docs)  # helper function not shown here
    query_engine = index.as_query_engine(similarity_top_k=top_k)
  
    # perform inference with RAG pipeline on a provided questions `eval_qs`
    pred_response_objs = get_responses(
        eval_qs, query_engine, show_progress=True
    )

    # perform evaluations of predictions by comparing them to reference
    # responses `ref_response_strs`
    evaluator = SemanticSimilarityEvaluator(...)
    eval_batch_runner = BatchEvalRunner(
        {"semantic_similarity": evaluator}, workers=2, show_progress=True
    )
    eval_results = eval_batch_runner.evaluate_responses(
        eval_qs, responses=pred_response_objs, reference=ref_response_strs
    )

    # get semantic similarity metric
    mean_score = np.array(
        [r.score for r in eval_results["semantic_similarity"]]
    ).mean()

    return RunResult(score=mean_score, params=params_dict)

# 2. Build ParamTuner object
param_dict = {"chunk_size": [256, 512, 1024]} # params/values to search over
fixed_param_dict = { # fixed hyperparams
  "top_k": 2,
    "docs": docs,
    "eval_qs": eval_qs[:10],
    "ref_response_strs": ref_response_strs[:10],
}
param_tuner = ParamTuner(
    param_fn=objective_function,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    show_progress=True,
)

# 3. Execute hyperparameter search
results = param_tuner.tune()
best_result = results.best_run_result
best_chunk_size = results.best_run_result.params["chunk_size"]
'''

# using cohere rerank for better retrieval 
'''
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor import LongLLMLinguaPostprocessor

### Recipe
### Define a Postprocessor object, here CohereRerank
### Build QueryEngine that uses this Postprocessor on retrieved docs

# Build CohereRerank post retrieval processor
api_key = os.environ["COHERE_API_KEY"]
cohere_rerank = CohereRerank(api_key=api_key, top_n=2)

# Build QueryEngine (RAG) using the post processor
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[cohere_rerank],
)

# Use your advanced RAG
response = query_engine.query(
    "What did Sam Altman do in this essay?"
)
'''