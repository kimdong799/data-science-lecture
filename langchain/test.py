from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-pro",# 모델 종류
                             convert_system_message_to_human=True, #ChatGoogleGenerativeAI각 SystemMessage를 지원하지 않아 해당 파라미터를 True로 전달하여 SystemMessage 사용 가능
                             temperature=0.2, # 창의성 정도 0~1
                            )

template = '{company}이 무슨 기업인지 간략하게 소개해줘'
prompt = PromptTemplate(template=template, input_variables=['company'])
output_parser = StrOutputParser()

# 연결된 체인(Chain)객체 생성
llm_chain = prompt | llm | output_parser

print(llm_chain.invoke('마이크로소프트'))