import os
import streamlit as st

from utils.llm_config import llm
from utils.process_document import process_pdf
from utils.process_vector import add_to_vector_store, load_existing_vector_store

persist_directory = 'db'

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
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
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