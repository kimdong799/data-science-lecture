from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import (
  HumanMessage,
  AIMessage 
)

from langchain_core.chat_history import (
  BaseChatMessageHistory,
  InMemoryChatMessageHistory,
)
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

response = with_message_history.invoke(
  [HumanMessage(content="Hi, I'm Bob")],
  config=config,
)
print(response.content)

response = with_message_history.invoke(
  [HumanMessage(content="What's my name?")],
  config=config,
)
print(response.content)