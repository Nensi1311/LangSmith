from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 

load_dotenv()

prompt = PromptTemplate.from_template("{question}")

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

model = ChatOpenAI(
    api_key=openrouter_api_key, 
    base_url="https://openrouter.ai/api/v1", 
    model="google/gemini-2.5-flash",
    model_kwargs={"max_tokens": 500} 
)
parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"question": "What is LangSmith?"})
print(response)