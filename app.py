import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# Load environment variables
load_dotenv()

# Optional: Enable LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# LLM: using Ollama with local deepseek model
llm = ChatOllama(model="deepseek-r1:1.5b")  # Make sure this matches your Ollama model name

# Output parser
output_parser = StrOutputParser()

# Chain the prompt → LLM → output
chain = prompt | llm | output_parser

# Streamlit UI
st.title("LangChain Chatbot with DeepSeek (Ollama)")
input_text = st.text_input("💬 Ask a question:")

if input_text:
    with st.spinner("Thinking..."):
        response = chain.invoke({'question': input_text})
        st.success("Here's the answer:")
        st.write("🤖", response)
