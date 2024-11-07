import os
import json
from pathlib import Path
import datetime
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, LLMChain
from langchain import PromptTemplate
from langchain.llms import OpenAI as LangChainOpenAI  # Correct import for OpenAI LLM

# Set the page config at the top of the script
st.set_page_config(layout="wide", page_title="سـند", page_icon="⚖️")

load_dotenv()

# Get the OpenAI API key from the environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
embedding = OpenAIEmbeddings()

# Custom CSS to move sidebar to the right
st.markdown("""
        <style>
        body {
            direction: rtl;
            text-align: right;
        }
        .stTextInput label {
            float: right;
        }
        </style>
        """, unsafe_allow_html=True)

# Streamlit app setup
with st.sidebar:
    st.image("/Users/remaalnssiry/Desktop/SAND/Logo.png", width=150)
    app_selection = st.sidebar.radio("أختر:", ["الباحث القانوني", "القضايا المشابهة"])

def app1():
    # Add custom CSS for chat messages
    st.markdown("""
    <style>
    # .stTextInput, .stMarkdown, .stButton {
    # float: right;
    # }
    .title {
        text-align: center;
        font-size: 2.5em;
        color: ##e0f7fa;
    }
    .chat-message-user, .chat-message-ai {
        display: flex;
        align-items: center;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    .chat-message-user {
        background-color: #e0f7fa;
        align-self: flex-end;
        justify-content: flex-end;
        color: #004d40;
        float: right;
    }
    .chat-message-user::before {
        content: "👤"; /* Custom icon for user */
        margin-left: 10px;
    }
    .chat-message-ai {
        background-color: #e8f5e9;
        align-self: flex-start;
        justify-content: flex-start;
        color: #2e7d32;
        float: left;
    }
    .chat-message-ai::before {
        content: "🤖"; /* Custom icon for AI */
        margin-right: 10px;

    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="title"> ⚖️ اسألني عن الأنظمة في السعودية</h1>', unsafe_allow_html=True)

    # Load documents
    directory = Path("/Users/remaalnssiry/Desktop/SAND/Legal_researcher/best_chatboot")
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

    # Initialize embedding and retriever
    embedding = OpenAIEmbeddings()
    retriever = Chroma(
        persist_directory="/Users/remaalnssiry/Desktop/SAND/Legal_researcher/docs2_best/chroma",
        embedding_function=embedding
    ).as_retriever()

    # Split documents
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1737,
        chunk_overlap=150,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)

    # Set model based on date
    current_date = datetime.datetime.now().date()
    llm_name = "gpt-3.5-turbo-0301" if current_date < datetime.date(2023, 9, 2) else "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=llm_name, temperature=0)

    # Set up QA chain with prompt
    template = """استخدم الأجزاء التالية من السياق للإجابة على السؤال في النهاية. إذا كنت لا تعرف الإجابة، قل فقط أنك لا تعرف، ولا تحاول اختلاق إجابة. استخدم ثلاث جمل كحد اقل. قل دائمًا "شكرًا على السؤال!" في نهاية الجواب.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-ai">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input field for user to type a message
    user_input = st.text_input("", key="user_input")
    if st.button("ارسل"):
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Retrieve relevant documents and generate response
            docs_ = retriever.get_relevant_documents(user_input)
            result = qa_chain({"query": user_input})
            answer = result["result"]
            st.session_state.messages.append({"role": "assistant", "content": answer})

            st.experimental_rerun()  # Rerun the app to display the new message

def app2():
    dataset_dir = "/Users/remaalnssiry/Desktop/SAND/dataset"

    llm = ChatOpenAI(temperature=0)
    map_template = """فيما يلي مجموعة من القضايا القانونيه تتعلق بحالات حدثت في المملكة العربية السعودية، بما في ذلك نص القضيه والحكم النهائي والتواريخ التي حدثت فيها،
    تم طمس اسماء الاشخاص وبعض الاماكن التي تخص كل قضية واستبدالها بـ `(...)` للحفاظ على خصوصية الجميع:
    '''
    {docs}
    '''
    استنادًا إلى قائمة المستندات هذه، يرجى تحديد المواضيع الرئيسية في كل حالة.
    """
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    reduce_template = """فيما يلي مجموعة من الملخصات لحالات تشمل الأحكام والتواريخ:
    {docs}
    يرجى أخذ هذه الملخصات واختصارها في ملخص نهائي وموحد للموضوعات الرئيسية.
    """
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="docs")
    reduce_documents_chain = ReduceDocumentsChain(combine_documents_chain=combine_documents_chain, collapse_documents_chain=combine_documents_chain, token_max=4000)
    map_reduce_chain = MapReduceDocumentsChain(llm_chain=map_chain, reduce_documents_chain=reduce_documents_chain, document_variable_name="docs", return_intermediate_steps=False)

    summarized_docs = []
    pdf_summaries = {}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    retriever = Chroma(persist_directory='/Users/remaalnssiry/Desktop/SAND/Similarity_search/embeding', embedding_function=embedding).as_retriever()

    # Update the path to the correct JSON file location
    json_file_path = '/Users/remaalnssiry/Desktop/SAND/Similarity_search/result_dict.json'

    # Load the JSON mapping file
    with open(json_file_path, 'r') as f:
        file_mapping = json.load(f)

    def search_similarities(keyword):
        top_docs = retriever.get_relevant_documents(keyword)
        return top_docs[:2]

    st.markdown("""
        <style>
        body {
            direction: rtl;
            text-align: right;
        }
        .stTextInput label {
            float: right;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("ابحث عن القضايا القانونية المتشابهة في المملكة العربية السعودية")

    user_query = st.text_area("يرجى كتابه سيناريو القضية القانونية التي ترغب في البحث عنها:")

    if st.button("بحث"):
        if user_query:
            most_similar_docs = search_similarities(user_query)
            st.write("المستندات الأكثر تشابهاً:")

            for i, doc in enumerate(most_similar_docs):
                file_path = doc.metadata['file_path']
                file_name = os.path.basename(file_path)
                drive_id = file_mapping.get(file_path.replace("/Users/remaalnssiry/Desktop/SAND/dataset", ""))

                if drive_id:
                    drive_link = f"https://drive.google.com/file/d/{drive_id}/view"
                    st.write(f"**القضيه المتشابه رقم {i+1}:**")
                    st.markdown(f" [ملف القضيه]({drive_link})", unsafe_allow_html=True)
                else:
                    st.write(f"**القضيه المتشابه رقم {i+1}:**")
                    st.write(f"**مسار الملف:** {file_path}")
                st.write(f"**تلخيص القضيه:** {doc.page_content}")

            final_summary = ""
            for doc in most_similar_docs:
                chunks = text_splitter.split_text(doc.page_content)
                split_doc_chunks = [
                    Document(page_content=chunk, metadata={'file_path': doc.metadata['file_path'], 'chunk_index': i})
                    for i, chunk in enumerate(chunks)
                ]
                summary = map_reduce_chain.run(split_doc_chunks)
                final_summary += summary + "\n\n"

            st.write("**الملخص النهائي:**")
            st.write(final_summary)
        else:
            st.write("الرجاء ادخال نص القضيه.")

# Display the selected app
if app_selection == "الباحث القانوني":
    app1()
elif app_selection == "القضايا المشابهة":
    app2()
