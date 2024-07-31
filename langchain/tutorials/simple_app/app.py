from dotenv import load_dotenv
from typing import List

from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes

# load settings
load_dotenv()

# define model
model = ChatGoogleGenerativeAI(model="gemini-pro",
                               convert_system_message_to_human=True,
                               )

# define template
system_template = "{company}가 어떤 사업으로 돈을 버는지 알려줘"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{company}")]
)

# define output parser
parser = StrOutputParser()

# chaining
chain = prompt_template | model | parser

# define app
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# adding chain route
add_routes(
  app,
  chain,
  path="/chain"
)

if __name__ == "__main__":
  import uvicorn
  
  uvicorn.run(app, host="localhost", port=8000)