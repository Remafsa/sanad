# !pip install openai
# !pip install -U langchain-community
# from langchain_community.document_loaders import TextLoader
# !pip install langchain
# !pip install python-dotenv


import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv

load_dotenv()

# Get the OpenAI API key from the environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

import json
from pathlib import Path
from pprint import pprint

# !pip install jq

import json
from pathlib import Path
from langchain_core.documents import Document


directory = Path("/content/drive/MyDrive/best_chatboot")

documents = []
for file_path in directory.glob('*.json'):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        metadata = {
            'الرابط': data['الرابط'],
            'الملف': str(file_path),
            'النظام': data['النظام'],
        }
        for chapter, articles in data['الفصول'].items():
            for article, content in articles.items():
                documents.append(Document(content, metadata={**metadata, 'الفصل': chapter, 'المادة': article}))

from langchain.text_splitter import TokenTextSplitter

# !pip install tiktoken

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

import os

directory = "/content/drive/MyDrive/best_chatboot"
filenames = []

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        filenames.append(os.path.join(directory, filename))

embedding1 = embedding.embed_documents(filenames)

from langchain.vectorstores import Chroma

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1737,
    chunk_overlap=150,
    length_function=len
)

splits = text_splitter.split_documents(documents)

# !pip install chromadb

retriever = Chroma(persist_directory="/content/drive/MyDrive/docs2_best/chroma", embedding_function=embedding).as_retriever()





import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)


from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"  # or any other chain type you want to use
)


################here ii

# from langchain.prompts import PromptTemplate

# # Build prompt
# template = """استخدم الأجزاء التالية من السياق للإجابة على السؤال في النهاية. إذا كنت لا تعرف الإجابة، قل فقط أنك لا تعرف، ولا تحاول اختلاق إجابة.  قل دائمًا "شكرًا على السؤال!" في نهاية الجواب.
# {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,  # Use the retriever you initialized
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
# )


#######USER INTERFACE###

# import streamlit as st
# st.title(" ⚖️اسألني عن الانظمة في السعودية")

# question = st.text_input("💬اكتب سؤالك هنا")
# if question:
#     result = qa_chain({"query": question})
#     answer = result["result"]
#     st.write(f"Answer: {answer}")

    # '''st.write("Source Documents:")
    # for doc in result['source_documents']:
    #     st.write(f"- {doc.metadata['الملف']} (Chapter: {doc.metadata['الفصل']}, Article: {doc.metadata['المادة']})")'''


# from langchain_community.chat_message_histories import (
#     StreamlitChatMessageHistory,
# )

# history = StreamlitChatMessageHistory(key="chat_messages")

# history.add_user_message("hi!")
# history.add_ai_message("whats up?")

# history.messages

# # Optionally, specify your own session_state key for storing messages
# msgs = StreamlitChatMessageHistory(key="special_app_key")

# if len(msgs.messages) == 0:
#     msgs.add_ai_message("How can I help you?")

# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_openai import ChatOpenAI

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are an AI chatbot having a conversation with a human."),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{question}"),
#     ]
# )

# chain = prompt | ChatOpenAI()


# chain_with_history = RunnableWithMessageHistory(
#     chain,
#     lambda session_id: msgs,  # Always return the instance created earlier
#     input_messages_key="question",
#     history_messages_key="history",
# )

# import streamlit as st

# for msg in msgs.messages:
#     st.chat_message(msg.type).write(msg.content)

# if prompt := st.chat_input():
#     st.chat_message("human").write(prompt)

#     # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
#     config = {"configurable": {"session_id": "any"}}
#     response = chain_with_history.invoke({"question": prompt}, config)
#     st.chat_message("ai").write(response.content)

##########################best UI is unser############
# import streamlit as st
# from langchain.prompts import PromptTemplate
# from langchain_community.chat_message_histories import StreamlitChatMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_openai import ChatOpenAI
# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
# #from langchain.retrievers import DummyRetriever  # Use a real retriever in production

# # Initialize chat message history
# msgs = StreamlitChatMessageHistory(key="special_app_key")

# if len(msgs.messages) == 0:
#     msgs.add_ai_message("كيف يمكنني مساعدتك؟")

# # Define chat prompt
# chat_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are an AI chatbot having a conversation with a human."),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{question}"),
#     ]
# )

# # Initialize the OpenAI chat model for conversation
# chat_llm = ChatOpenAI()

# # Combine the chat prompt with the model
# chat_chain = chat_prompt | chat_llm

# # Wrap the chain with message history handling
# chain_with_history = RunnableWithMessageHistory(
#     chat_chain,
#     lambda session_id: msgs,
#     input_messages_key="question",
#     history_messages_key="history",
# )

# # Define the QA prompt template
# template = """استخدم الأجزاء التالية من السياق للإجابة على السؤال في النهاية. إذا كنت لا تعرف الإجابة، قل فقط أنك لا تعرف، ولا تحاول اختلاق إجابة. استخدم ثلاث جمل كحد أقصى. اجعل الإجابة مختصرة قدر الإمكان. واجب بالترحيب في بداية المحادثة وفي نهاية كل جواب قل انا هنا دائما لخدمتك." في نهاية الجواب.
# {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# # Set up your LLM and retriever (dummy retriever for example purposes)
# llm = OpenAI()  # Replace with your actual LLM
# #retriever = DummyRetriever()  # Replace with your actual retriever

# retriever = Chroma(persist_directory="/content/drive/MyDrive/docs2_best/chroma", embedding_function=embedding).as_retriever()
# # Initialize the retrieval-based QA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
# )

# # Streamlit interface
# st.title('⚖️اسألني عن الانظمة في السعودية')

# # Display chat history
# for msg in msgs.messages:
#     st.chat_message(msg.type).write(msg.content)

# # Input field for user to type a message
# if user_input := st.chat_input():
#     st.chat_message("user").write(user_input)
#     msgs.add_user_message(user_input)

#     # Generate response using the retrieval-based QA chain
#     result = qa_chain({"query": user_input})
#     answer = result["result"]
#     st.chat_message("ai").write(answer)
#     msgs.add_ai_message(answer)


import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
#from langchain.retrievers import DummyRetriever  # Use a real retriever in production

# Initialize chat message history
msgs = StreamlitChatMessageHistory(key="special_app_key")

if len(msgs.messages) == 0:
    msgs.add_ai_message("كيف يمكنني مساعدتك؟")

# Define chat prompt
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Initialize the OpenAI chat model for conversation
chat_llm = ChatOpenAI()

# Combine the chat prompt with the model
chat_chain = chat_prompt | chat_llm

# Wrap the chain with message history handling
chain_with_history = RunnableWithMessageHistory(
    chat_chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Define the QA prompt template
template = """استخدم الأجزاء التالية من السياق للإجابة على السؤال في النهاية. إذا كنت لا تعرف الإجابة، قل فقط أنك لا تعرف، ولا تحاول اختلاق إجابة.واذا سال عن حالك قل انا بخير واجب بالترحيب في بداية المحادثة وفي نهاية كل جواب قل انا هنا دائما لخدمتك..
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Set up your LLM and retriever (dummy retriever for example purposes)
llm = OpenAI()  # Replace with your actual LLM
#retriever = DummyRetriever()  # Replace with your actual retriever

retriever = Chroma(persist_directory="/content/drive/MyDrive/docs2_best/chroma", embedding_function=embedding).as_retriever()
# Initialize the retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# Add custom CSS for chat messages
st.markdown("""
<style>

.title {
    text-align: center;
}

.chat-message-user {
    display: flex;
    align-items: center;
    background-color: #82C42E;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}
.chat-message-user::before {
    content: "👤"; /* Custom icon for user */
    margin-right: 10px;

}

.chat-message-ai {
    display: flex;
    align-items: center;
    background-color: #9EA45C;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}
.chat-message-ai::before {
    content: "🤖"; /* Custom icon for AI */
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)

st.image("https://e3.365dm.com/23/09/768x432/skynews-judge-stock_6285365.jpg?20230915095121", use_column_width=True)

# Streamlit interface
#st.title('⚖️اسألني عن الانظمة في السعودية')
st.markdown('<h1 class="title">⚖️ اسألني عن الأنظمة في السعودية</h1>', unsafe_allow_html=True)


# Display chat history
for msg in msgs.messages:
    if msg.type == "user":
        st.markdown(f'<div class="chat-message-user">{msg.content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message-ai">{msg.content}</div>', unsafe_allow_html=True)

# Input field for user to type a message
if user_input := st.chat_input():
    st.markdown(f'<div class="chat-message-user">{user_input}</div>', unsafe_allow_html=True)
    msgs.add_user_message(user_input)

    docs_ = retriever.get_relevant_documents(user_input)
    # Generate response using the retrieval-based QA chain
    result = qa_chain({"query": user_input})
    answer = result["result"]
    st.markdown(f'<div class="chat-message-ai">{answer}</div>', unsafe_allow_html=True)
    msgs.add_ai_message(answer)
