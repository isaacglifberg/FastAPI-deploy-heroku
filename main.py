from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import os
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
import uvicorn


app = FastAPI()

class UserQuery(BaseModel):
    question: str



app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://schoolsoft-support-bot-fe9057e7d85a.herokuapp.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



script_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(script_dir, 'my_vectorstore.index')


embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


def get_conversation_chain():
    llm = ChatOpenAI()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    )
    return conversation_chain

conversation_chain = get_conversation_chain()

# Error handling middleware
@app.exception_handler(HTTPException)
async def http_exception_handler(exc, request):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def generic_exception_handler(ecx, request):
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.post("/ask/")
async def ask_question(query: UserQuery):
    try:
        input_data = {
            'question': query.question,
            'chat_history': ''
        }
        result = conversation_chain.invoke(input_data)
        return {"result": result.get('answer', 'No answer found.')}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

