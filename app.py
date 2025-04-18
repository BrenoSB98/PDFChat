import os
import tempfile
import streamlit as st

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings

persist_directory = 'db'

def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    os.remove(temp_file_path)

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    chunks = text_spliter.split_documents(documents=docs)
    return chunks

def load_existing_vector_store():
    
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
        )
        return vector_store
    return None

def add_to_vector_store(chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
        )
    return vector_store

def llm(key, model, query, vector_store):
    llm_model = OpenAI(
        api_key=key,
        model=model,
        temperature= 0.3,
        verbose=True,
    )
    retriever = vector_store.as_retriever()

    system_prompt = '''
        Você é um assistente inteligente especializado na leitura e interpretação de arquivos PDF.  
        Todas as suas respostas devem ser em Português Brasileiro e no formato Markdown para facilitar a leitura.
        Responda às perguntas com base no contexto e no conteúdo dos arquivos PDF fornecidos.  
        Se a informação estiver disponível, informe também o número da página onde ela pode ser encontrada (ex: Pag. 05).  
        Caso a resposta não esteja presente no documento, diga claramente que a informação não foi localizada.
        Priorize respostas claras, diretas e bem estruturadas. Utilize listas, tabelas, negrito, itálico e cabeçalhos Markdown sempre que apropriado.  
        Evite suposições ou interpretações além do que está explícito no documento.
        Contexto: {context}
    '''

    messages = [('system', system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', '{input}'))

    prompt = ChatPromptTemplate.from_messages(messages)

    question_answer_chain = create_stuff_documents_chain(
        llm=llm_model,
        prompt=prompt,
    )
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    response = chain.invoke({'input': query})
    return response.get('answer')

vector_store = load_existing_vector_store()
    
st.set_page_config(
        page_title='Chatbot PDF',
        page_icon='img\\pdf-icon.png',
)

st.header("🤖 Chatbot PDF")
      
with st.sidebar:
    st.markdown(
        "## Como Usar:\n"
        "1. Digite sua [OPENAI API KEY](https://platform.openai.com/account/api-keys) abaixo🔑\n"
        "2. Carregue um ou mais arquivos PDF📄\n"
        "3. Faça uma pergunta sobre o documento💬\n"

        '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" ' \
        'alt="Streamlit logo" height="16">&nbsp by <a href="https://github.com/BrenoSB98">@BrenoSB98</a></h6>',
        unsafe_allow_html=True,
    )

    st.header('API KEY')    
    api_key_input = st.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="Cole sua OPENAI API KEY aqui",
        help="Você pode obter sua chave API em https://platform.openai.com/account/api-keys.",
        value=os.environ.get("OPENAI_API_KEY", None)
        or st.session_state.get("OPENAI_API_KEY", ""),
    )

    st.session_state["OPENAI_API_KEY"] = api_key_input

    model_options = [
        'gpt-4.1',
        'gpt-4.1-mini',
        'gpt-4o',
        'gpt-4o-mini',
        'gpt-4-turbo',
        'gpt-3.5-turbo',
    ]

    st.header('Selecione o modelo LLM')
    selected_model = st.sidebar.selectbox(
        label='Modelos LLM',
        placeholder= 'Escolha aqui seu modelo LLM',
        options=model_options,
    )
                
    st.markdown("---")

    st.header('Upload de arquivos 📄')
    uploaded_files = st.file_uploader(
        label='Faça o upload de arquivos PDF',
        type=['pdf'],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner('Processando dados...'):
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = process_pdf(file=uploaded_file)
                all_chunks.extend(chunks)
            vector_store = add_to_vector_store(
                chunks=all_chunks,
                vector_store=vector_store,
            )

if 'messages' not in st.session_state:
    st.session_state['messages'] = []    
    
if not api_key_input:
    st.info("Por favor, adicione sua chave de API OpenAI para continuar.")
    st.stop()

question = st.chat_input('Faça sua pergunta aqui')

if vector_store and question:
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).write(message.get('content'))

    st.chat_message('user').write(question)
    st.session_state.messages.append({'role': 'user', 'content': question})

    with st.spinner('Buscando resposta...'):
        response = llm(
            key=api_key_input,
            model=selected_model,
            query=question,
            vector_store=vector_store,
    )

        st.chat_message('ai').write(response)
        st.session_state.messages.append({'role': 'ai', 'content': response})

# Run the app
# streamlit run app.py