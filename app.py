import pandas as pd
from tqdm import tqdm
import pinecone
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import streamlit as st
import openai


# connect to pinecone environment
pinecone.init(
    api_key="d4f20339-fcc1-4a11-b04f-3800203eacd2",
    environment="us-east1-gcp"  
)

index_name = "abstractive-question-answering"

index = pinecone.Index(index_name)

# Initialize models from HuggingFace

@st.cache_resource
def get_t5_model():
    return pipeline("summarization", model="t5-base", tokenizer="t5-base")

@st.cache_resource
def get_flan_t5_model():
    return pipeline("summarization", model="google/flan-t5-base", tokenizer="google/flan-t5-base")
    
@st.cache_resource
def get_embedding_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=device)
    model.max_seq_length = 512
    return model
    
@st.cache_data()
def save_key(api_key):
    return api_key

retriever_model = get_embedding_model()

def query_pinecone(query, top_k, model):
    # generate embeddings for the query
    xq = model.encode([query]).tolist()
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc

def format_query(query_results):
    # extract passage_text from Pinecone search result
    context = [result['metadata']['merged_text'] for result in query_results['matches']]
    return context

def gpt3_summary(text):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=text+"\n\nTl;dr",
    temperature=0.1,
    max_tokens=512,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=1
    )
    return response.choices[0].text

def gpt3_qa(query, answer):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Q: " + query + "\nA: " + answer,
    temperature=0,
    max_tokens=512,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\n"]
    )
    return response.choices[0].text
  
  
st.title("Abstractive Question Answering - APPL")

query_text = st.text_input("Input Query", value="Who is the CEO of Apple?")

num_results = int(st.number_input("Number of Results to query", 1, 5, value=2))

query_results = query_pinecone(query_text, num_results, retriever_model)

context_list = format_query(query_results)



# Choose decoder model

models_choice = ["GPT3 (text_davinci)", "GPT3 - QA", "T5", "FLAN-T5"]

decoder_model = st.selectbox(
    'Select Decoder Model',
    models_choice)

st.subheader("Answer:")


if decoder_model == "GPT3 (text_davinci)":
  openai_key = st.text_input("Enter OpenAI key")
  api_key = save_key(openai_key)
  openai.api_key = api_key
  output_text = []
  for context_text in context_list:
    output_text.append(gpt3_summary(context_text))
  generated_text = " ".join(output_text)
  st.write(gpt3_summary(generated_text))

elif decoder_model=="GPT3 - QA":
  openai_key = st.text_input("Enter OpenAI key")
  api_key = save_key(openai_key)
  openai.api_key = api_key
  output_text = []
  for context_text in context_list:
    output_text.append(gpt3_qa(query_text, context_text))
  generated_text = " ".join(output_text)
  st.write(gpt3_qa(query_text, generated_text))

elif decoder_model == "T5":
  t5_pipeline = get_t5_model()
  output_text = []
  for context_text in context_list:
    output_text.append(t5_pipeline(context_text)[0]["summary_text"])
  generated_text = " ".join(output_text)
  st.write(t5_pipeline(generated_text)[0]["summary_text"])
  
elif decoder_model == "FLAN-T5":
  flan_t5_pipeline = get_flan_t5_model()
  output_text = []
  for context_text in context_list:
    output_text.append(flan_t5_pipeline(context_text)[0]["summary_text"])
  generated_text = " ".join(output_text)
  st.write(flan_t5_pipeline(generated_text)[0]["summary_text"])

st.subheader("Retrieved Text:")

for context_text in context_list:
  st.markdown(f"- {context_text}")

