import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

# ## LangSmith Tracking ##
# Only use these if you have a LangSmith API key
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with GOOGLE"


# ## Prompt Template ##
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)


def generate_response(question, api_key, llm, temperature, max_tokens):

    google_api_key = api_key

    llm = ChatGoogleGenerativeAI(
        model=llm,
        google_api_key=google_api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_level="low"
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    answer = chain.invoke({
        "question": question
    })

    return answer


# ## Title ##
st.title("Enhanced Q&A Chatbot with GOOGLE")


# ## Side Bar ##
st.sidebar.title("Settings")

api_key = st.sidebar.text_input(
    "Enter your Google AI API Key:",
    type="password"
)


# ## Drop down to select various models ##
llm = st.sidebar.selectbox(
    "Select an AI model",
    [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro"
    ]
)


# ## Adjust response ##
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=100,
    max_value=4000,
    value=1000
)


# ## Main interface ##
st.write("Go ahead and ask any question")

user_input = st.text_input("You:")


if user_input:

    if not api_key:
        st.warning("Please enter your Google Gemini API key.")

    else:
        response = generate_response(
            user_input,
            api_key,
            llm,
            temperature,
            max_tokens
        )

        st.write(response)

else:
    st.write("Please provide the query")