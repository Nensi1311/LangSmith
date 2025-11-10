from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

os.environ['LANGCHAIN_PROJECT'] ='Sequential Chain Demo'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following report in bullet points:\n\n{report}',
    input_variables=['report']
)

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

model1 = ChatOpenAI(
    api_key=openrouter_api_key, 
    base_url="https://openrouter.ai/api/v1", 
    model="google/gemini-2.5-flash",
    model_kwargs={"max_tokens": 500},
    temperature=0.7
)

model2 = ChatOpenAI(
    api_key=openrouter_api_key, 
    base_url="https://openrouter.ai/api/v1", 
    model="openai/gpt-4o-mini",
    model_kwargs={"max_tokens": 500},
    temperature=0.5
)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'run_name': 'sequential chain',
    'tags': ['llm app', 'report generation', 'summarization'],
    'metadata': {'model1': 'google/gemini-2.5-flash', 'model2': 'openai/gpt-4o-mini', 'model1_temp': 0.7, 'model2_temp': 0.5, 'parser': 'stroutputparser'}
}

response = chain.invoke({"topic": "The impact of AI on modern education"}, config=config)
print(response)