import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
import requests
from bs4 import BeautifulSoup

# 데이터 로드 및 전처리
@st.cache_resource
def load_data():
    url = 'https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/resolve/main/bitext-telco-llm-chatbot-training-dataset.csv'
    df = pd.read_csv(url)
    return df

# 모델 및 토크나이저 로드
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "EleutherAI/polyglot-ko-5.8b"  # 한국어 지원 대형 언어 모델
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Sentence Transformer 모델 로드
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# 번역기 초기화
translator = Translator()

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

# 메인 함수
def main():
    st.title("Telco Chatbot")

    df = load_data()
    tokenizer, model = load_model_and_tokenizer()
    sentence_transformer = load_sentence_transformer()

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        # RAG
        relevant_context = retrieve_relevant_context(user_input, df, sentence_transformer)
        
        if not relevant_context.empty:
            context = " ".join(relevant_context['instruction'] + " " + relevant_context['response'])
            input_text = context + " " + user_input
            source = "Fine-tuning Data"
        else:
            # 웹 검색
            web_results = web_search(user_input)
            if web_results:
                input_text = " ".join(web_results) + " " + user_input
                source = "tworld 페이지"
            else:
                input_text = user_input
                source = "웹서치"

        # 생성
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        
        with torch.no_grad():
            output = model.generate(input_ids, 
                                    attention_mask=attention_mask, 
                                    max_length=150, 
                                    num_return_sequences=1, 
                                    no_repeat_ngram_size=2, 
                                    top_k=50, 
                                    top_p=0.95, 
                                    temperature=0.7)

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        st.write("챗봇 응답:")
        st.write(response)
        st.write(f"위 답변은 {source}를 참고했습니다.")

if __name__ == "__main__":
    main()
