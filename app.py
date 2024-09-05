import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Tuple

st.set_page_config(page_title="텔코 고객센터 챗봇", page_icon="🤖")

@st.cache_resource
def load_data() -> pd.DataFrame:
    df_csv = load_dataset("bitext/Bitext-telco-llm-chatbot-training-dataset", split="train").to_pandas()[['instruction', 'response']].rename(columns={"response": "output"}).sample(50, random_state=42)
    df_parquet = load_dataset("akshayjambhulkar/customer-support-telecom-alpaca", split="train").to_pandas()[['instruction', 'output']].sample(50, random_state=42)
    return pd.concat([df_csv, df_parquet], ignore_index=True)

@st.cache_resource
def load_model() -> SentenceTransformer:
    return SentenceTransformer('all-MiniLM-L6-v2')

def find_most_similar(query: str, df: pd.DataFrame, model: SentenceTransformer) -> Tuple[str, float]:
    query_embedding = model.encode([query])
    instruction_embeddings = model.encode(df['instruction'].tolist())
    similarities = np.dot(query_embedding, instruction_embeddings.T)[0]
    most_similar_idx = similarities.argmax()
    return df.iloc[most_similar_idx]['output'], similarities[most_similar_idx]

def main():
    st.title("텔코 고객센터 챗봇 🤖")
    st.write("고객센터 관련 질문을 입력하세요:")

    df = load_data()
    model = load_model()

    question = st.text_input("질문을 입력하세요:")
    
    if question:
        with st.spinner('답변을 생성 중입니다...'):
            answer, similarity = find_most_similar(question, df, model)
            st.write(f"챗봇 답변: {answer}")
            st.write(f"유사도: {similarity:.2f}")

if __name__ == "__main__":
    main()
