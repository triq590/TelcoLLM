import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

# RAG 함수
def retrieve_relevant_context(query, df, sentence_transformer):
    query_embedding = sentence_transformer.encode([query])
    df['embedding'] = df['instruction'].apply(lambda x: sentence_transformer.encode(x))
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(query_embedding, x.reshape(1, -1))[0][0])
    return df.nlargest(3, 'similarity')

# 응답 생성 함수
def generate_response(query, context, model, tokenizer):
    input_text = context + " " + query
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
    return response

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
            response = generate_response(user_input, context, model, tokenizer)
            source = "Fine-tuning Data"
        else:
            response = "죄송합니다. 해당 질문에 대한 정확한 답변을 찾지 못했습니다. 고객센터(114)로 문의해 주시면 자세히 안내해 드리겠습니다."
            source = "기본 응답"

        st.write("챗봇 응답:")
        st.write(response)
        st.write(f"위 답변은 {source}를 참고했습니다.")

if __name__ == "__main__":
    main()
