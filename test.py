import os
from pydantic import BaseModel
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# 환경 변수에서 API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 요청 모델 정의
class QuestionRequest(BaseModel):
    question: str

def smart_qa(question: str) -> str:
    try:
        # GPT-3.5 초기화
        general_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY
        )

        # GPT-3.5용 프롬프트 템플릿
        general_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """당신은 친절한 한국어 AI 어시스턴트입니다.
                주어진 질문에 답할 수 있다면 답변하고,
                문서나 추가 컨텍스트가 필요하다고 판단되면
                "NEED_CONTEXT"라고 말하세요. 다시 한번 강조합니다.
                주어진 질문에 제대로 답 못할 것 같으면
                차라리 "NEED_CONTEXT"라고 답하세요"""
            ),
            ("human", "{question}")
        ])

        # GPT-3.5로 먼저 시도
        chain = general_prompt | general_llm
        result = chain.invoke({"question": question})
        initial_response = result.content
        print(initial_response)

        # "NEED_CONTEXT"가 있으면 추가 문서 기반 응답
        if "NEED_CONTEXT" in initial_response:
            splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=600,
                chunk_overlap=100,
                length_function=len
            )

            # 문서 로드 및 분할
            loader = TextLoader("files/청년자립정보.txt", encoding='utf-8')
            documents = loader.load()
            docs = splitter.split_documents(documents)

            # 임베딩 및 벡터 저장소 생성
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vectorstore = FAISS.from_documents(docs, embeddings)

            # 검색기 설정
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            # 문서 검색용 프롬프트
            doc_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """다음 컨텍스트를 사용하여 질문에 답하세요:
                                    {context}"""
                ),
                ("human", "{question}"),
            ])

            # 문서 검색 체인 실행
            doc_chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough()
                }
                | doc_prompt
                | general_llm  # 기존 LLM을 사용합니다.
            )

            doc_result = doc_chain.invoke(question)
            return doc_result.content

        return initial_response
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"

# 예시 질문
question = "구름톤 팀원에 대해 소개해줘"
answer = smart_qa(question)
print("응답:", answer)