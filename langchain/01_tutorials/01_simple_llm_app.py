from typing import List

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from langserve import add_routes

from dotenv import load_dotenv
import os

# load .env
load_dotenv()

# prompt template 생성
system_template = "다음의 언어로 번역해줘 {language}"
prompt_template = ChatPromptTemplate.from_messages([
  ('system', system_template),
  ('user', '{text}')
])

# 모델 생성
model = GoogleGenerativeAI(model="gemini-1.5-flash")

# 파서 생성
parser = StrOutputParser()

# 체인 생성
chain = prompt_template | model | parser

# App 정의
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="Simple API server using Langchain's Runnable interfaces"
)

# 체인 루트 추가
add_routes(
  app,
  chain,
  path="/chain"
)

if __name__ == "__main__":
  import uvicorn
  
  uvicorn.run(app, host="localhost", port=8000)