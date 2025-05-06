from dotenv import load_dotenv
load_dotenv()

from run_evals import send_to_openai  
import autogen  #A
import asyncio  #B
import os
from typing_extensions import Annotated
from azure.search.documents.aio import SearchClient as asyncSearchClient  #C
from azure.core.credentials import AzureKeyCredential
import logging
import sys
from openai import AsyncOpenAI  #A


logging.basicConfig(level=logging.DEBUG)

config_list = [
  {
    "model": "gpt-4o",
    "api_key": os.environ.get("OPENAI_API_KEY"),  #A
  }
]

def construct_azure_ai_search_assistant_agent():
  print("Constructing search assistant...")
  azure_ai_search_assistant_agent = autogen.AssistantAgent(  #A
    name="search_assistant",
    system_message="""You are a helpful assistant for a company called Products, Inc.  #B
You have access to an Azure AI Search Index containing product records, and you may search them. The correct syntax for a search is: "what you want to search for". Please use the search function to find enough information to answer the user's question. DO NOT rely on your own knowledge, ONLY use the information retrieved from the search. If you only find one document, that is ok. If the document contains very little information, that is ok. Please be honest about what you find. I am not looking for perfection, just the truth. 
You are amazing and you can do this. I will pay you $200 for an excellent result, but only if you follow all instructions exactly.""",
    llm_config={  #C
      "config_list": config_list,
      "temperature": 0,  
      "stream": False,
    },
    code_execution_config=False,
  )
  return azure_ai_search_assistant_agent  #D


def construct_azure_ai_search_executor_agent():
  print("Constructing search executor...")
  azure_ai_search_executor_agent = autogen.UserProxyAgent(  #A
  name="search_executor",  #B
  code_execution_config=False,
    system_message="""When enough information has been retrieved to answer the user's question to full satisfaction, please return "TERMINATE" to end the conversation. If more information must be collected, please return CONTINUE.""",  #D
    human_input_mode="NEVER",  #E
    max_consecutive_auto_reply=4,  #F
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),  #G
    llm_config={  #H
    "config_list": config_list,
    "temperature": 0.0,  #I
    "stream": False,  #J
})
  return azure_ai_search_executor_agent



product_db_assistant = construct_azure_ai_search_assistant_agent()
product_db_executor = construct_azure_ai_search_executor_agent()

async def get_query_embedding(query):  #A
  print("getting embedding...")
  async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
  embedding_response = await async_openai_client.embeddings.create(  #B
input=query, 
model="text-embedding-3-small",  #C
dimensions=1536  #D
)  
  return embedding_response.data[0].embedding  #

from azure.search.documents.models import VectorizedQuery
@product_db_assistant.register_for_execution()  #A
@product_db_executor.register_for_llm(  #A
  description="Search an Azure AI Search Index containing product documents like sales catalogs and user manuals."
)
async def search_product_documents(  #B
  search_term: Annotated[str, "Search term to search for."]
) -> str:
  print("searching documents...")

  loop = asyncio.get_event_loop()  #C
  search_client = asyncSearchClient(  #D
    endpoint=os.environ.get("AI_SEARCH_ENDPOINT"),
    index_name=os.environ.get("AI_SEARCH_NAME"),
    credential=AzureKeyCredential(os.environ.get("AI_SEARCH_KEY")),
  )

  

  query_embedding = await get_query_embedding(search_term)  #E
  vector_query = VectorizedQuery(  #F
    vector=query_embedding, k_nearest_neighbors=3, fields="Vector"
  )
  async with search_client:
    results = await search_client.search(
      search_text=search_term,  #G
      vector_queries=[vector_query],  #G
      top=3,
    )
    return [result async for result in results]

# search_results = asyncio.run(search_product_documents("Hasty Pants"))
# print(search_results)

def create_groupchat_and_manager(agents, groupchat_manager_name):
  print("Creating groupchat...")
  groupchat = autogen.GroupChat(
    agents=agents, messages=[], max_round=4, speaker_selection_method="round_robin"  #A
  )
  groupchat_manager = autogen.GroupChatManager(
    groupchat=groupchat,
    name=groupchat_manager_name,
    llm_config={"config_list": config_list, "stream": False},
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,  #B
  )
  return groupchat_manager

def construct_writer_agent():
  print("Building writer agent...")
  writer_assistant_agent = autogen.AssistantAgent( # A
    name="writer_assistant",
    system_message="""You are a helpful assistant for a company called Products, Inc.  #B
Your job is to answer the user's question using the provided information.
DO NOT rely on your own knowledge, ONLY use the provided info.
If you don't know the answer, just say you don't know. 
You are amazing and you can do this. I will pay you $200 for an excellent result, but only if you follow all instructions exactly.""",
    llm_config={ # C
    "config_list": config_list,
   "temperature": 0,
    "stream": True,
    },
    code_execution_config=False,
    max_consecutive_auto_reply=1,
    )
  return writer_assistant_agent

def check_language(user_question, answer):
    print("Checking lanugage...")
    check_language_prompt = f"""
    You are an expert at languages and translation.
    Please make sure the answer is written in the same language as the user's question.
    If the answer and the question are both written in the same language, return
    TRUE.
    Otherwise, return the answer translated into the same language as the user's question.
    
    For example, if the user's question is in German but the answer is in English, please
     return the answer, translated into German.
     
    User question: {user_question}
    
    Answer: {answer}"""    #A

    result = send_to_openai(check_language_prompt)
    result = result.content

    if "true" in result.lower():    #B
        return "LANGUAGE VERIFIED"
    else:
        return result    #C

import datetime

def store_answer_info(user_email, user_question, agents_search_results, final_answer):
  import uuid
  print("Storing answer info...")
  from azure.identity import DefaultAzureCredential
  from azure.cosmos import CosmosClient

  credential = DefaultAzureCredential()

  client = CosmosClient(url="https://ragchat.documents.azure.com:443/", credential=credential)
  database = client.get_database_client("ragchat_info")
  container = database.get_container_client("ragchat_logs")

  timestamp = str(datetime.datetime.now())  #B

  new_log = {
    "id": str(uuid.uuid4()),
    "user_email": user_email, 
    "user_question": user_question, 
    "agents_search_results": agents_search_results, 
    "final_answer": final_answer, 
    "timestamp": timestamp
    }  #C

  created_item = container.upsert_item(new_log)

async def RAGChat(chat_history, user_question, user_email, iostream):
  print("Running Ragchat...")
  user = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
     ) # A

  product_db_groupchat = create_groupchat_and_manager(
    [product_db_assistant, product_db_executor], 
    "product_db_groupchat"
    ) # B

  search_prompt = f"""Please search and find enough information to answer the user's question.
User Question: {user_question}""" # C

  async_chat_plan = [
    {
      "chat_id": 1,
      "recipient": product_db_groupchat,
      "message": search_prompt,
      "summary_method": "reflection_with_llm",
      "silent": True,
    }
  ] # D

  await user.a_initiate_chats(async_chat_plan) # E

  retrieved_data = product_db_assistant.chat_messages  #F
  
  writer_agent = construct_writer_agent()

  writer_prompt = f"""Please write the final answer to the user's question: \n{user_question}\n\n
  You may use the chat history to help you write the answer. \n {chat_history}\n\n
The information retrieved from the search agents is:
{retrieved_data}. I will tip you $200 for an excellent result."""

  iostream.print("ANSWER:")  #G

  writer_userproxy = autogen.UserProxyAgent(
    name="WriterUserproxy", # no spaces in the agent name
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content","").find("TERMINATE") >= 0,
  )  #H

  writer_userproxy.initiate_chat(
    recipient=writer_agent, message=writer_prompt, silent=True
  )  #I
  writer_chat_logs = writer_agent.chat_messages  #J
  logs_list = [writer_chat_logs[k] for k in writer_chat_logs.keys()][0]  #J

  if len(logs_list[-1]["content"]) <= 2:  #K
      final_answer = logs_list[-2]["content"]
  else:
      final_answer = logs_list[-1]["content"]

  translated_answer = check_language(user_question, final_answer)    #L
  if translated_answer=="LANGUAGE VERIFIED":
    pass
  else:
    final_answer = translated_answer
    iostream.print(final_answer)    

  # store_answer_info(user_email, user_question, str(retrieved_data), final_answer)    #M

triage_prompt = """You are a helpful assistant.  You are responsible for    
    categorizing user questions.    #A

If the user's question is about a product, please return *PRODUCT.    #B
For example, if the user asks a question about Dubious Parenting Advice, please return

    *PRODUCT 

If the user's question is about an order, please return *ORDER.    #C
For example, if the user asks a question about Order number 12345, 
please return

    *ORDER

If you are not sure which category a user's question belongs to, return 
*CLARIFY followed by a request for clarification in
square brackets.  Your request should try to gain enough information 
from the user to decide which of the above 2 categories you should  
choose for their question.    #D

    For example, if the user enters:

    12345689

    Please return:

*CLARIFY [I'm sorry but I don't understand what you are asking.  Are 
you looking for a product or an order?]

Remember that you ONLY have access to information in our Products and 
Orders databases.  If the user asks for information which would 
not be in either of those databases, please let them know that you do 
not have access to that information.    #E

    For example, if the user enters:
    What is the address of our headquarters?

    Please return:

*CLARIFY [I'm sorry but I don't have access to that information.  I 
only have access to information in our Products and Orders databases.  
If the information you are looking for is not in one of those two 
databases, then I don't have access to it.]    #F

If you cannot answer the user's question, please try to guide the user 
to a question that you can answer using the sources you have access to.

User Question: {}
Chat history: {}
    """

def triage(user_question, chat_history):    #A  
    print("Triaging question...")
    formatted_triage_prompt = triage_prompt.format(user_question, chat_history)    #B
    result = send_to_openai(formatted_triage_prompt)    #C
    result = result.content

    if "*PRODUCT" in result:    #D
        return "PRODUCT", result.strip()

    elif "*ORDER" in result:
        return "ORDER", result.strip()

    elif "*CLARIFY" in result:
        result = result.replace("[", "")
        result = result.replace("]", "")    #E
        result = result.replace("*CLARIFY", "")
        return "CLARIFY", result.strip()


from websockets.sync.client import connect as ws_connect
import autogen
from autogen.io.websockets import IOWebsockets  #A
import json

def on_connect(iostream: IOWebsockets) -> None:
  received_request = json.loads(iostream.input(), strict=False)  #A

  chat_history = received_request.get("chat_history")  #B
  user_email = received_request.get("user_email")  #B
  if len(chat_history) > 4:
    chat_history = chat_history[-4:]  #C
  user_question = chat_history[-1]  #D
  if len(user_question) > 1000:
    iostream.print("Sorry, your question is too long.")  #E
    # iostream.print("TERMINATE")  #H

    return

  loop = asyncio.new_event_loop()  #F
  asyncio.set_event_loop(loop)  #F

  question_category, message = triage(user_question, chat_history)

  if question_category == "PRODUCT":

    try:
      RAGChat_result = loop.run_until_complete(
        RAGChat(chat_history, user_question, user_email, iostream)
      )  #G
    finally:
      # iostream.print("TERMINATE")  #H
      loop.close()  #H
  elif question_category == "CLARIFY":
    iostream.print(message)
    # iostream.print("TERMINATE")  #H
    loop.close()  #H

import time
import logging

def main():
   
    if os.name == "nt":
      http_port = 8080
      asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    else:
       http_port = 8000

    with IOWebsockets.run_server_in_thread( # A
        host="0.0.0.0",on_connect=on_connect, port=http_port # B
) as uri: # C
        print(f"Websocket server running on {uri}")
        while True:
            time.sleep(0.01)

if __name__ == "__main__":
   main()