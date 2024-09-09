import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

# 데이터 로드 및 전처리
@st.cache_resource
def load_data():
    url = 'https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/resolve/main/bitext-telco-llm-chatbot-training-dataset.csv'
    df = pd.read_csv(url)
    return df

# Sentence Transformer 모델 로드
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# RAG 함수
def retrieve_relevant_context(query, df, sentence_transformer):
    query_embedding = sentence_transformer.encode([query])
    df['embedding'] = df['instruction'].apply(lambda x: sentence_transformer.encode(x))
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(query_embedding, x.reshape(1, -1))[0][0])
    return df.nlargest(3, 'similarity')

# 웹 검색 함수
def web_search(query):
    url = "https://www.tworld.co.kr/web/home"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 메뉴 항목 찾기 (실제 웹사이트 구조에 따라 수정 필요)
    menu_items = soup.find_all('a', class_='menu-item')
    
    relevant_items = []
    for item in menu_items:
        if query.lower() in item.text.lower():
            relevant_items.append(item.text)
    
    return relevant_items

# 간단한 응답 생성 함수
def generate_response(context, query):
    # 여기에 간단한 규칙 기반 응답 로직을 구현합니다
    if "요금제" in query:
        return "요금제 변경은 고객센터(114)로 문의하시거나 T world 앱에서 직접 변경하실 수 있습니다."
    elif "문의" in query:
        return "더 자세한 정보는 고객센터(114)로 문의해 주시기 바랍니다."
    else:
        return f"죄송합니다. '{query}'에 대한 정확한 정보를 제공하기 어렵습니다. 고객센터(114)로 문의해 주시면 자세히 안내해 드리겠습니다."

# 메인 함수
def main():
    st.title("Telco Chatbot")

    df = load_data()
    sentence_transformer = load_sentence_transformer()

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        # RAG
        relevant_context = retrieve_relevant_context(user_input, df, sentence_transformer)
        
        if not relevant_context.empty:
            context = " ".join(relevant_context['instruction'] + " " + relevant_context['response'])
            source = "Fine-tuning Data"
        else:
            # 웹 검색
            web_results = web_search(user_input)
            if web_results:
                context = " ".join(web_results)
                source = "tworld 페이지"
            else:
                context = user_input
                source = "웹서치"

        # 응답 생성
        response = generate_response(context, user_input)
        
        st.write("챗봇 응답:")
        st.write(response)
        st.write(f"위 답변은 {source}를 참고했습니다.")

if __name__ == "__main__":
    main()
