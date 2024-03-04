import streamlit as st
from urllib.request import urlopen
from langchain.vectorstores import ElasticsearchStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import Bedrock
from langchain.chains import RetrievalQA
import boto3
import json
from dataclasses import dataclass
from typing import Literal
import os
import streamlit.components.v1 as components

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str

def load_css():
    with open("D:\Project\saleary project\example\static\styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "retriever" not in st.session_state:
        AWS_ACCESS_KEY = st.secrets['AWS_ACCESS_KEY']
        AWS_SECRET_KEY = st.secrets['AWS_SECRET_KEY']
        AWS_REGION = st.secrets['AWS_REGION']

        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )

        CLOUD_ID = st.secrets['ELASTIC_SEARCH_ID']
        CLOUD_USERNAME = "elastic"
        CLOUD_PASSWORD = st.secrets['ELASTIC_PASSWORD']

        vector_store = ElasticsearchStore(
            es_cloud_id=CLOUD_ID,
            es_user=CLOUD_USERNAME,
            es_password=CLOUD_PASSWORD,
            index_name="seahawk5",
            strategy=ElasticsearchStore.SparseVectorRetrievalStrategy()
        )

        default_model_id = "amazon.titan-text-express-v1"
        AWS_MODEL_ID = default_model_id
        llm = Bedrock(
            client=bedrock_client,
            model_id=AWS_MODEL_ID
        )

        retriever = vector_store.as_retriever()

        st.session_state.retriever = RetrievalQA.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
def llm_response_temp(human_prompt):
    response = st.session_state.retriever(human_prompt)
    answer = response['result']
    source = []
    for doc in response["source_documents"]:
        source = doc.metadata['title']

    return answer, source

def on_click_callback():
    human_prompt = st.session_state.human_prompt
    llm_response, llm_source = llm_response_temp(human_prompt)
    st.session_state.history.append(
        Message("human", human_prompt)
    )
    st.session_state.history.append(
        Message("ai", llm_response)
    )


load_css()
initialize_session_state()

st.title("Seahawk 2024 Draft AI Tool")

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
credit_card_placeholder = st.empty()

with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
<div class="chat-row 
    {'' if chat.origin == 'ai' else 'row-reverse'}">
    <img class="chat-icon" src="app/static/{
        'Seahawks_Chatbot_Logo.png' if chat.origin == 'ai' 
                      else 'user_icon.png'}"
         width=32 height=32>
    <div class="chat-bubble
    {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
        {chat.message}
</div>
        """
        st.markdown(div, unsafe_allow_html=True)
    
    for _ in range(3):
        st.markdown("")

with prompt_placeholder:
    st.markdown("**Chat**")
    cols = st.columns((6, 1))
    cols[0].text_input(
        "Chat",
        value="Hello bot",
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit", 
        type="primary", 
        on_click=on_click_callback, 
    )

credit_card_placeholder.caption(f"""
Used {st.session_state.token_count} tokens""")

components.html("""
<script>
const streamlitDoc = window.parent.document;

const buttons = Array.from(
    streamlitDoc.querySelectorAll('.stButton > button')
);
const submitButton = buttons.find(
    el => el.innerText === 'Submit'
);

streamlitDoc.addEventListener('keydown', function(e) {
    switch (e.key) {
        case 'Enter':
            submitButton.click();
            break;
    }
});
</script>
""", 
    height=0,
    width=0,
)
