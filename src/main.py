# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:45:56 2023

@author: Vijay Budhewar
"""


import openai
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import json
from langchain.llms import OpenAI
from streamlit_lottie import st_lottie
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
import streamlit as st
import requests
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header




openai_config='openai_config.json'  #give standard path
open_ai_config = open(openai_config)
openai_configuration=json.load(open_ai_config)
os.environ['OPENAI_API_KEY']=openai_configuration['key']
openai.api_key=openai_configuration['key']

embeddings = OpenAIEmbeddings()

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

    
def get_store_faiss_goals(text_samples,action):
    print('Current working directory is ', os.getcwd())
    persist_directory_faiss = os.path.join(os.getcwd(),"FAISS")
    print('FAISS directory',persist_directory_faiss)
    if action == 'store':
        db_faiss = FAISS.from_documents(text_samples, embeddings)
        db_faiss.save_local(persist_directory_faiss)
    if action == 'get_db':
        db_faiss = FAISS.from_documents(text_samples, embeddings)
        return(db_faiss)
    if action == 'get':
        new_db = FAISS.load_local(persist_directory_faiss, embeddings)
    #Storing of FAISS can be done using pickle file
        retriever_faiss = new_db.as_retriever(search_kwargs={"k":3},reduce_k_below_max_tokens=True)
        return retriever_faiss
    
retriever_faiss = get_store_faiss_goals('' , 'get')




condense_prompt = PromptTemplate.from_template(
    ("""Here is the chat history or conversation of You and user {chat_history}
    Currently , the question from user is ({question}) , make sure you primarily remember all the details from user like name , age, other details also most importaatly about the website
    please generate a fianl question based on this information using personal information given by user 
    """)
)

combine_docs_custom_prompt = PromptTemplate.from_template(
    (""" You are representing as a chatbot for a Massage parlour website, this website is for providing services of massages 
      As a marketing consultant for this website , be always polite and give assisting responses by using the following question : {question}
     from user based on below data : {context} \n'
     Answer :

     """)
)
        
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = ConversationalRetrievalChain.from_llm(OpenAI(model_name='text-davinci-003',max_tokens=200,temperature=0),
    retriever=retriever_faiss, # see below for vectorstore definition,
    memory = memory,
    condense_question_prompt=condense_prompt,
    combine_docs_chain_kwargs = dict(prompt=combine_docs_custom_prompt)
)



with st.sidebar:
   
    lotti_sidebar=load_lottieurl('https://lottie.host/0a821b57-194f-4227-a39f-5f6d841179f0/J2yuT4lMnT.json')
    st_lottie(lotti_sidebar,reverse=True,height=300,  width=300,speed=1,  loop=True,quality='high')
    st.title("This is a Project for Business Chatbot")
    
    st.markdown('''This application showcases the capabilities of AI using OpenAI's LLMs
    ''')
    
ai_assistnt,sample_tab= st.tabs(["AI Chatbot","Description"])

def get_answer(user_input):
    ai_response = chain({"question": f"{user_input}"})['answer']
    return ai_response


with ai_assistnt:
    
    if 'generated' not in st.session_state:
        print('Inside')
        st.session_state['generated'] = ["I'm your AI Assistant, How may I help you?"]
    
    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hi!']
        
    def get_text():
        input_text = st.text_input("You: ", "", key="input")
        #st.session_state["You: "] = ""
        return input_text
    
    
    def generate_response(prompt):
        response = get_answer(prompt)
        return response
    
    def resp():
        with response_container:
            if user_input:
                response = generate_response(user_input)
               
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)
                
            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state['generated'][i], key=str(i))

    
    response_container = st.container()
    user_input=None                
    input_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    with input_container:
        user_input = get_text()
        resp()





