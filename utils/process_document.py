# import os
# import tempfile

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader

# def process_pdf(file):
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#         temp_file.write(file.read())
#         temp_file_path = temp_file.name

#     loader = PyPDFLoader(temp_file_path)
#     docs = loader.load()

#     os.remove(temp_file_path)

#     text_spliter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=400,
#     )
#     chunks = text_spliter.split_documents(documents=docs)
#     return chunks