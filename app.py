# app.py

# 1. 필요한 라이브러리 설치 (필요한 경우)
!pip install -U pyarrow==15.0.0
!pip install -U sentence-transformers langchain langchain_community langchain_chroma datasets openai transformers streamlit tokenizers==0.13.3

# 2. 라이브러리 import 및 환경 설정
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

# OpenAI API 키 설정 (Streamlit secrets 사용)
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

# 3. 데이터 로드 (샘플링된 데이터 사용으로 테스트 시간 단축)
df_csv = load_dataset("bitext/Bitext-telco-llm-chatbot-training-dataset", split="train").to_pandas()[['instruction', 'response']].rename(columns={"response": "output"}).sample(50, random_state=42)
df_parquet = load_dataset("akshayjambhulkar/customer-support-telecom-alpaca", split="train").to_pandas()[['instruction', 'output']].sample(50, random_state=42)
df = pd.concat([df_csv, df_parquet], ignore_index=True)

# 4. Fine-tuning 준비
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

# 패딩 토큰 설정 (필수)
tokenizer.pad_token = tokenizer.eos_token

# Fine-tuning에 사용할 데이터셋 준비
def tokenize_function(examples):
    # 입력(input)과 출력(label)을 각각 토큰화
    inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(examples['label'], padding="max_length", truncation=True, max_length=128)

    # labels의 패딩 토큰을 -100으로 처리하여 학습 시 무시되도록 설정
    labels["input_ids"] = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels["input_ids"]]
    
    inputs["labels"] = labels["input_ids"]
    return inputs

# train_data에서 instruction과 output을 input과 label로 변환
train_data = df[['instruction', 'output']].rename(columns={"instruction": "input", "output": "label"}).to_dict('records')

# Hugging Face Dataset 변환
from datasets import Dataset
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))

# 토큰화 (열 이름을 'input'과 'label'로 변경)
tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

# 텐서로 변환할 때 리스트의 길이를 모두 맞춰야 함
# 이를 위해 입력 데이터와 출력 데이터의 길이가 같은지 확인하고 일관되게 처리
def collate_fn(batch):
    input_ids = torch.tensor([example['input_ids'] for example in batch], dtype=torch.long)
    labels = torch.tensor([example['labels'] for example in batch], dtype=torch.long)
    return {'input_ids': input_ids, 'labels': labels}

# TrainingArguments 설정 (평가 생략)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    evaluation_strategy="no",  # 평가를 생략하여 오류 해결
    logging_dir="./logs",
    save_total_limit=2,
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=collate_fn  # 리스트 길이를 맞춰서 학습할 수 있게 해줍니다.
)

# Fine-tuning 수행
trainer.train()

# Fine-tuned 모델 저장
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# 5. 임베딩 및 벡터 스토어 생성
docs = [f"질문: {row['instruction']}\n답변: {row['output']}" for _, row in df.iterrows()]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_text("\n\n".join(docs))

# Document 객체로 변환
documents = [Document(page_content=split) for split in splits]

# HuggingFace 임베딩 모델 사용
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Chroma Vector Store 생성
vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model)

# 검색기 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 6. Streamlit 앱 구성 (챗봇 웹 실행)
st.title("고객센터 챗봇")
st.write("고객센터 관련 질문을 입력하세요:")

# 사용자 입력 받기
question = st.text_input("질문을 입력하세요:")

# Prompt 생성
prompt = PromptTemplate.from_template("""
당신은 통신사 고객 서비스 담당자입니다. 다음의 맥락을 참고하여 질문에 답변해주세요. 
모르는 내용이라면 모른다고 솔직히 말씀해주세요. 
최대 3문장으로 간결하게 답변해주시고, 고객에게 친절하고 공손한 어조로 대답해주세요.

질문: {question}
맥락: {context}
답변:
""")

# RAG 체인 생성
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

# 질문에 대한 응답 처리 및 결과 출력
if question:
    result = rag_chain({"query": question})
    st.write(f"챗봇 답변: {result['result']}")
