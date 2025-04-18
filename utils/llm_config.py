import streamlit as st

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def llm(key, model, query, vector_store):
    llm_model = ChatOpenAI(
        api_key=key,
        model=model,
        temperature= 0.5,
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
