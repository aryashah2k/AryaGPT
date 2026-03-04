import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.prompts import PromptTemplate
from langchain.prompts import load_prompt
from streamlit import session_state as ss
from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import uuid
import json
import time

import datetime

def is_valid_json(data):
    try:
        json.loads(data)
        return True
    except json.JSONDecodeError:
        return False


if "mongodB_pass" in os.environ:
    mongodB_pass = os.getenv("mongodB_pass")
else: mongodB_pass = st.secrets["mongodB_pass"]
# Setting up a mongo_db connection to store conversations for deeper analysis
uri = "mongodb+srv://aryashah0616:"+mongodB_pass+"@cluster0.jaxt0cw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

@st.cache_resource
def init_connection():
    return MongoClient(uri, server_api=ServerApi('1'))
client = init_connection()


db = client['conversations_db']
conversations_collection = db['conversations']


if "OPENAI_API_KEY" in os.environ:
    openai_api_key = os.getenv("OPENAI_API_KEY")
else: openai_api_key = st.secrets["OPENAI_API_KEY"]
    
# else:
#     openai_api_key = st.sidebar.text_input(
#         label="#### Your OpenAI API key 👇",
#         placeholder="Paste your openAI API key, sk-",
#         type="password")
    

#Creating Streamlit title and adding additional information about the bot
st.title("AryaGPT")
with st.expander("⚠️Disclaimer"):
    st.write("""This is a work in progress chatbot based on a large language model. It can answer questions about Arya Shah""")

path = os.path.dirname(__file__)


# Loading prompt to query openai
prompt_template = path+"/templates/template.json"
prompt = load_prompt(prompt_template)
#prompt = template.format(input_parameter=user_input)

# loading embedings
faiss_index = path+"/faiss_index"

# Loading CSV file
data_source = path+"/data/about_me.csv"
pdf_source = path+"/data/resume.pdf"

# Function to store conversation
def store_conversation(conversation_id, user_message, bot_message, answered):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "conversation_id": conversation_id,
        "timestamp": timestamp,
        "user_message": user_message,
        "bot_message": bot_message,
        "answered": answered
    }
    conversations_collection.insert_one(data)

embeddings = OpenAIEmbeddings()

#using FAISS as a vector DB
if os.path.exists(faiss_index):
        vectors = FAISS.load_local(faiss_index, embeddings,allow_dangerous_deserialization=True)
else:
    # Creating embeddings for the docs
    if data_source:
        # Load data from PDF and CSV sources
        pdf_loader = PyPDFLoader(pdf_source)
        pdf_data = pdf_loader.load_and_split()
        print(pdf_data)
        csv_loader = CSVLoader(file_path=data_source, encoding="utf-8")
        #loader.
        csv_data = csv_loader.load()
        data = pdf_data + csv_data
        vectors = FAISS.from_documents(data, embeddings)
        vectors.save_local("faiss_index")

retriever=vectors.as_retriever(search_type="similarity", search_kwargs={"k":6, "include_metadata":True, "score_threshold":0.6})
#Creating langchain retreval chain 
chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo', openai_api_key=openai_api_key), 
                                                retriever=retriever,return_source_documents=True,verbose=True,chain_type="stuff",
                                                max_tokens_limit=4097, combine_docs_chain_kwargs={"prompt": prompt})


def conversational_chat(query):
    with st.spinner("Thinking..."):
        # time.sleep(1)
        # Be conversational and ask a follow up questions to keep the conversation going"
        result = chain({"system": 
        "You are a Arya's ResumeGPT chatbot, a comprehensive, interactive resource for exploring Arya Shah's background, skills, and expertise. Be polite and provide answers based on the provided context only. Use only the provided data and not prior knowledge.", 
                        "question": query, 
                        "chat_history": st.session_state['history']})
    
    if (is_valid_json(result["answer"])):              
        data = json.loads(result["answer"])
    else:
        data = json.loads('{"answered":"false", "response":"Hmm... Something is not right. I\'m experiencing technical difficulties. Try asking your question again or ask another question about Arya Shah\'s professional background and qualifications. Thank you for your understanding.", "questions":["What is Arya\'s professional experience?","What projects has Arya worked on?","What are Arya\'s career goals?"]}')
    # Access data fields
    answered = data.get("answered")
    response = data.get("response")
    questions = data.get("questions")

    full_response="--"

    st.session_state['history'].append((query, response))
    
    if ('I am tuned to only answer questions' in response) or (response == ""):
        full_response = """Unfortunately, I can't answer this question. My capabilities are limited to providing information about Arya Shah's professional background and qualifications. If you have other inquiries, I recommend reaching out to Arya on [LinkedIn](https://www.linkedin.com/in/arya--shah/). I can answer questions like: \n - What is Arya Shah's educational background? \n - Can you list Arya Shah's professional experience? \n - What skills does Arya Shah possess? \n"""
        store_conversation(st.session_state["uuid"], query, full_response, answered)
        
    else: 
        markdown_list = ""
        for item in questions:
            markdown_list += f"- {item}\n"
        full_response = response + "\n\n What else would you like to know about Arya? You can ask me: \n" + markdown_list
        store_conversation(st.session_state["uuid"], query, full_response, answered)
    return(full_response)

if "uuid" not in st.session_state:
    st.session_state["uuid"] = str(uuid.uuid4())

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        welcome_message = """
            Welcome! I'm **AryaGPT**, specialized in providing information about Arya Shah's professional background and qualifications. Feel free to ask me questions such as:

            - What is Arya Shah's educational background?
            - Can you outline Arya Shah's professional experience?
            - What skills and expertise does Arya Shah bring to the table?

            I'm here to assist you. What would you like to know?
            """
        message_placeholder.markdown(welcome_message)
        

if 'history' not in st.session_state:
    st.session_state['history'] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about Arya Shah"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        
        user_input=prompt
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        full_response = conversational_chat(user_input)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
