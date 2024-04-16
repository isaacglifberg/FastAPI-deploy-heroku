from PyPDF2 import PdfReader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from docx import Document
from dotenv import load_dotenv
import os
from langchain_text_splitters import CharacterTextSplitter



def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            print(f"Failed to process {pdf}: {e}")
    return text

def get_text_from_docx(filename):
    try:
        document = Document(filename)
        return '\n'.join(para.text for para in document.paragraphs)
    except Exception as e:
        print(f"Failed to read {filename}: {e}")
        return ""

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def preprocess_documents_and_store_embeddings(documents):
    all_text_chunks = []

    for document in documents:
        if document.endswith('.pdf'):
            text = get_pdf_text([document])
        elif document.endswith('.docx'):
            text = get_text_from_docx(document)
        text_chunks = get_text_chunks(text)
        all_text_chunks.extend(text_chunks)  

    if all_text_chunks:
        vectorstore = get_vectorstore(all_text_chunks)
        save_path = '/Users/isaacglifberg/Desktop/chatbot_fastAPI/backend/my_vectorstore.index'
        vectorstore.save_local(save_path)


def find_documents(root_folder, extensions=('.pdf', '.docx')):
    documents = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(extensions):
                documents.append(os.path.join(subdir, file))
    return documents

documents = find_documents('/Users/isaacglifberg/Desktop/test_documents/OneDrive_1_2024-04-12')
preprocess_documents_and_store_embeddings(documents)


