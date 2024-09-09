import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_data():
    url = 'https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/resolve/main/bitext-telco-llm-chatbot-training-dataset.csv'
    df = pd.read_csv(url)
    return df.sample(n=100, random_state=42)  # 100개 샘플링

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

def retrieve_relevant_context(query, df, sentence_transformer):
    query_embedding = sentence_transformer.encode([query])
    df['embedding'] = df['instruction'].apply(lambda x: sentence_transformer.encode(x))
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(query_embedding, x.reshape(1, -1))[0][0])
    return df.nlargest(3, 'similarity')

def main():
    st.title("Telco Chatbot")

    df = load_data()
    sentence_transformer = load_sentence_transformer()

    user_input = st.text_input("질문을 입력하세요:")

    if user_input:
        relevant_context = retrieve_relevant_context(user_input, df, sentence_transformer)
        
        if not relevant_context.empty:
            response = relevant_context.iloc[0]['response']
            source = "Fine-tuning Data"
        else:
            response = "죄송합니다. 해당 질문에 대한 정확한 답변을 찾지 못했습니다."
            source = "기본 응답"

        st.write("챗봇 응답:")
        st.write(response)
        st.write(f"위 답변은 {source}를 참고했습니다.")

if __name__ == "__main__":
    main()
