# app.py

import os
import pandas as pd
import streamlit as st
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

# Streamlit 설정
st.set_page_config(page_title="텔코 고객센터 챗봇", page_icon="🤖")

# OpenAI API 키 설정 (Streamlit secrets 사용)
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

@st.cache_resource
def load_data():
    df_csv = load_dataset("bitext/Bitext-telco-llm-chatbot-training-dataset", split="train").to_pandas()[['instruction', 'response']].rename(columns={"response": "output"}).sample(50, random_state=42)
    df_parquet = load_dataset("akshayjambhulkar/customer-support-telecom-alpaca", split="train").to_pandas()[['instruction', 'output']].sample(50, random_state=42)
    return pd.concat([df_csv, df_parquet], ignore_index=True)

@st.cache_resource
def fine_tune_model(df):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=128)
        labels = tokenizer(examples['label'], padding="max_length", truncation=True, max_length=128)
        labels["input_ids"] = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels["input_ids"]]
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_data = df[['instruction', 'output']].rename(columns={"instruction": "input", "output": "label"}).to_dict('records')
    train_dataset = load_dataset('dict', data={'train': train_data})['train']
    tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        evaluation_strategy="no",
        logging_dir="./logs",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    return model, tokenizer

@st.cache_resource
def create_vectorstore(df):
    docs = [f"질문: {row['instruction']}\n답변: {row['output']}" for _, row in df.iterrows()]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_text("\n\n".join(docs))
    documents = [Document(page_content=split) for split in splits]
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=documents, embedding=embedding_model)

def main():
    st.title("텔코 고객센터 챗봇 🤖")
    st.write("고객센터 관련 질문을 입력하세요:")

    df = load_data()
    model, tokenizer = fine_tune_model(df)
    vectorstore = create_vectorstore(df)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template("""
    당신은 통신사 고객 서비스 담당자입니다. 다음의 맥락을 참고하여 질문에 답변해주세요. 
    모르는 내용이라면 모른다고 솔직히 말씀해주세요. 
    최대 3문장으로 간결하게 답변해주시고, 고객에게 친절하고 공손한 어조로 대답해주세요.

    질문: {question}
    맥락: {context}
    답변:
    """)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    question = st.text_input("질문을 입력하세요:")
    if question:
        with st.spinner('답변을 생성 중입니다...'):
            result = rag_chain({"query": question})
            st.write(f"챗봇 답변: {result['result']}")

if __name__ == "__main__":
    main()
