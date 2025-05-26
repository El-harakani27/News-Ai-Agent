import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START,END,StateGraph
from langgraph.prebuilt import tools_condition,ToolNode
from langgraph.graph.message import add_messages
from langchain.tools import tool
from typing_extensions import TypedDict
from typing import Annotated
from langchain_tavily import TavilySearch
import os
from groq import Groq
from io import BytesIO
from dotenv import load_dotenv
from transformers import pipeline


load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# class State(TypedDict):
#     messages : Annotated[list,add_messages]


# st.title("News Agent")
# audio_value = st.audio_input("Press to record an Audio")
# if audio_value:

#     audio_bytes = audio_value.read()
#     audio_buffer = BytesIO(audio_bytes)
#     audio_buffer.name = "name.m4a"
#     client = Groq()
#     transcription = client.audio.transcriptions.create(
#       file=audio_buffer,
#       model="whisper-large-v3",
#       response_format="verbose_json",
#     )
#     st.subheader("Transcription")
#     st.write(transcription.text)
      
    # llm = ChatGroq(model="qwen-qwq-32b")
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system","You are a transelator that is going to translate input text this text can be in any language but you translate to english"),
    #     ("user","{input}")
    # ]
    # )
    # chain = prompt | llm | StrOutputParser()
    # output = chain.invoke({"input":transcription.text})
    # st.subheader("Translation")
    # st.write(output.split('</think>')[-1])

# class State(TypedDict):
#     messages : Annotated[list,add_messages]
# build_graph = StateGraph(State)

# # Initialize the LLM and tools
# tool = TavilySearch(max_results=2)
# tools = [tool]
# llm = ChatGroq(model="qwen-qwq-32b")
# llm_with_tools = llm.bind_tools(tools)

# # Define the chatbot function
# def chatbot(state: State):
#     return {"messages":[llm_with_tools.invoke(state['messages'])]}
# build_graph.add_node("chatbot",chatbot)
# build_graph.add_node("tool",ToolNode(tools=[tool]))
# build_graph.add_conditional_edges(
#     "chatbot",
#     tools_condition,
# )
# # Any time a tool is called, we return to the chatbot to decide the next step
# build_graph.add_edge("tools", "chatbot")
# build_graph.add_edge(START, "chatbot")
# graph = build_graph.compile()
# i = input("Please type:")
# graph.invoke({"messages":[{"role":"system","content":"Summarize the input that you got with the language provided to you"},
#                           {"role":"user","content":{i}}]})
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages
from langchain.tools import tool
from typing_extensions import TypedDict
from typing import Annotated
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv

# ---- Load environment variables ----
load_dotenv()

# ---- Define State ----
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ---- Define Prompt ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the input that you got with the language provided to you."),
    ("user", "{input}")
])

# ---- Initialize LLM and Tools ----
tool = TavilySearch(max_results=2)
tools = [tool]
llm = ChatGroq(model="qwen-qwq-32b")
llm_with_tools = llm.bind_tools(tools)

# ---- Define Chatbot Node ----
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ---- Build Graph ----
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools=tools))

# Define conditional routing from chatbot to tools
builder.add_conditional_edges("chatbot", tools_condition, {"tools"})
builder.add_edge("tools", "chatbot")
builder.set_entry_point("chatbot")
graph = builder.compile()

# ---- Streamlit UI ----
st.set_page_config(page_title="LangGraph Chatbot", layout="wide")
st.title("🧠 News Agent")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "Summarize the input that you got with the language provided to you."}
    ]

def input_to_agent(user_input):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        # Run through LangGraph
        result = graph.invoke({"messages": st.session_state.chat_history})
        response_message = result["messages"][-1]

        st.session_state.chat_history.append({"role":"assistant","content":response_message.content})
# User Input
audio =st.audio_input("Please enter an audio")
if audio:
    audio_read = audio.read()
    audio_buffer = BytesIO(audio_read)
    audio_buffer.name = "audio.m4a"
    client = Groq()
    transcription = client.audio.transcriptions.create(
      file=audio_buffer,
      model="whisper-large-v3",
      response_format="verbose_json",
  )
    

    st.success("Transcription Complete")
    st.write(f"**Transcription:** {transcription.text}")
    input_to_agent(transcription.text)


user_input = st.chat_input("Type your message...")
input_to_agent(user_input)

# print(st.session_state.chat_history)
# Chat Display
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])
