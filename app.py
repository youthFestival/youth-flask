import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

from langchain_teddynote import logging

logging.langsmith("youth-chatbot")

# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("data/festival_list.pdf")
docs = loader.load()

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

# 단계 3: 임베딩(Embedding) 생성
embeddings = OpenAIEmbeddings()

# 단계 4: DB 생성(Create DB) 및 저장
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
    Based on the provided context, answer the questions.
    If you are unsure or do not know the answer, respond with "I don't know." Write your answers in Korean.
    If the information searched includes a URL, include it separated by a newline from the rest of the text in the response.
    Provide accurate answers.


#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
)

# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 단계 8: 체인(Chain) 생성
chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)


# question = "고려대 축제 언제하는지 알려줘"
# response = chain.invoke(question)
# print(response)

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def hello():
    return "hello"


@app.route('/chatbot', methods=['POST'])
def generate_response():
    data = request.json
    question = data['question']
    # RAG 체인을 사용하여 답변 생성
    result = chain.invoke(question)
    answer = result

    # 링크 추출 (정규 표현식을 사용하여 링크 추출)
    link_pattern = r'http?://[^\s\()]+'
    match = re.search(link_pattern, answer)
    link = match.group(0) if match else None

    # JSON 형식으로 응답 및 링크 분리
    response_json = {
        "answer": answer,
        "link": link
    }

    return jsonify(response_json)


if __name__ == '__main__':
    app.run(port=5050, debug=True)
