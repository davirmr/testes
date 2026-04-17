import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model="gpt-oss-20b:free",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
     max_tokens=100
)

while True:
    pergunta = input("Pergunta: ")
    
    if pergunta.lower() == "sair":
        break

    resposta = llm.invoke([
    HumanMessage(content=f"Responda de forma curta e objetiva: {pergunta}")
    ])
    
    print("Resposta:", resposta.content)