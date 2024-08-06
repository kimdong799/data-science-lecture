from dotenv import load_dotenv
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# load settings
load_dotenv()

# define model
model = ChatGoogleGenerativeAI(model="gemini-pro",
                               convert_system_message_to_human=True,
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id] = InMemoryChatMessageHistory()
  return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)

config = {"configurable":
    {"session_id": "ABC"}
  }

trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    HumanMessage(content="안녕 난 고길동이야! 곧 유학을 떠날 예정인데 회화좀 도와줄래?"),
    AIMessage(content="괜찮습니다! 제가 도와드릴게요!"),
    HumanMessage(content="외국인 친구 역할로 대답해줘!"),
    AIMessage(content="네 좋아요! 최선을 다해 도와드릴게요."),
]

trimmer.invoke(messages)

prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      "{language}로 먼저 대답해주고 한국어로 무슨 뜻인지 설명해줘."
    ),
    MessagesPlaceholder(variable_name="messages"),
  ]
)

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

# 대화 1
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="안녕 만나서 반가워! 내 이름이 뭐라고?")],
        "language": "English",
    }
)
print(response.content)

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="내 이름이 뭐라고~?")],
        "language": "English",
    },
    config=config,
)

# Streaming
for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="내 첫 외국인 친구야! 앞으로 친하게 지내자!")],
        "language": "English",
    },
    config=config,
):
    print(r.content, end="|")