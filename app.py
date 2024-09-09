import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

# 데이터 로드 및 전처리
@st.cache_resource
def load_data():
    url = 'https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/resolve/main/bitext-telco-llm-chatbot-training-dataset.csv'
    df = pd.read_csv(url)
    return df

# 모델 및 토크나이저 로드
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "skt/kogpt2-base-v2"  # 한국어 GPT-2 모델
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
    df['embedding'] = df['question'].apply(lambda x: sentence_transformer.encode(x))
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(query_embedding, x.reshape(1, -1))[0][0])
    return df.nlargest(3, 'similarity')

# Fine-tuning 함수 (실제 환경에서는 별도로 실행하고 결과만 로드해야 함)
def fine_tune_model(model, tokenizer, df):
    # 이 부분은 실제 fine-tuning 로직을 구현해야 합니다.
    # 여기서는 간단한 예시만 제공합니다.
    st.write("실제 시나리오에서는 여기서 Fine-tuning이 수행됩니다.")
    return model, tokenizer

# 메인 함수
def main():
    st.title("텔코 고객센터 챗봇 데모")

    df = load_data()
    tokenizer, model = load_model_and_tokenizer()
    sentence_transformer = load_sentence_transformer()

    # Fine-tuning (실제로는 이 부분을 별도로 실행하고 결과를 저장해야 함)
    model, tokenizer = fine_tune_model(model, tokenizer, df)

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        # 한국어 질문을 영어로 번역
        translated_input = translator.translate(user_input, src='ko', dest='en').text

        # RAG
        relevant_context = retrieve_relevant_context(translated_input, df, sentence_transformer)
        context = " ".join(relevant_context['question'] + " " + relevant_context['answer'])

        # 컨텍스트를 한국어로 번역
        translated_context = translator.translate(context, src='en', dest='ko').text

        # 생성
        input_text = translated_context + " " + user_input
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

    # 데이터 프레임 표시 (디버깅 목적)
    st.subheader("샘플 데이터")
    st.dataframe(df.head())

if __name__ == "__main__":
    main()
