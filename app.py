import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Tuple

st.set_page_config(page_title="텔코 고객센터 챗봇", page_icon="🤖")

@st.cache_resource
def load_data() -> pd.DataFrame:
    try:
        df_csv = load_dataset("bitext/Bitext-telco-llm-chatbot-training-dataset", split="train").to_pandas()[['instruction', 'response']].rename(columns={"response": "output"}).sample(50, random_state=42)
        df_parquet = load_dataset("akshayjambhulkar/customer-support-telecom-alpaca", split="train").to_pandas()[['instruction', 'output']].sample(50, random_state=42)
        return pd.concat([df_csv, df_parquet], ignore_index=True)
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {str(e)}")
        return pd.DataFrame(columns=['instruction', 'output'])

@st.cache_resource
def load_model() -> SentenceTransformer:
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {str(e)}")
        return None

def find_most_similar(query: str, df: pd.DataFrame, model: SentenceTransformer) -> Tuple[str, float]:
    try:
        query_embedding = model.encode([query])
        instruction_embeddings = model.encode(df['instruction'].tolist())
        similarities = np.dot(query_embedding, instruction_embeddings.T)[0]
        most_similar_idx = similarities.argmax()
        return df.iloc[most_similar_idx]['output'], similarities[most_similar_idx]
    except Exception as e:
        st.error(f"유사도 계산 중 오류 발생: {str(e)}")
        return "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다.", 0.0

def main():
    st.title("텔코 고객센터 챗봇 🤖")
    st.write("고객센터 관련 질문을 입력하세요:")

    df = load_data()
    model = load_model()

    if df.empty or model is None:
        st.error("챗봇을 초기화하는 데 문제가 발생했습니다. 나중에 다시 시도해 주세요.")
        return

    question = st.text_input("질문을 입력하세요:")
    
    if question:
        with st.spinner('답변을 생성 중입니다...'):
            answer, similarity = find_most_similar(question, df, model)
            st.write(f"챗봇 답변: {answer}")
            st.write(f"유사도: {similarity:.2f}")

    st.sidebar.markdown("### 챗봇 정보")
    st.sidebar.info("이 챗봇은 텔코 고객센터 문의에 대한 답변을 제공합니다. 실제 상담원과 대화하는 것이 아니며, 일반적인 정보만 제공합니다.")

if __name__ == "__main__":
    main()
