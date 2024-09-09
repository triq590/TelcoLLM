import streamlit as st
import pandas as pd

# 데이터 다운로드 및 로드
@st.cache_data
def load_data():
    url = 'https://huggingface.co/datasets/bitext/Bitext-telco-llm-chatbot-training-dataset/resolve/main/bitext-telco-llm-chatbot-training-dataset.csv'
    df = pd.read_csv(url)
    return df.sample(n=100, random_state=42)  # 100개 샘플 추출

# 데이터 로드
df = load_data()

# Streamlit 앱
st.title("텔코 고객센터 챗봇 데모")

# 사용자 입력
user_input = st.text_input("질문을 입력하세요:")

if user_input:
    # 간단한 키워드 매칭 (실제 프로덕션에서는 더 복잡한 로직이 필요합니다)
    matching_rows = df[df['question'].str.contains(user_input, case=False, na=False)]
    
    if not matching_rows.empty:
        st.write("답변:")
        st.write(matching_rows.iloc[0]['answer'])
    else:
        st.write("죄송합니다. 해당 질문에 대한 답변을 찾을 수 없습니다.")

# 데이터 프레임 표시 (디버깅 목적)
st.subheader("샘플 데이터")
st.dataframe(df)
