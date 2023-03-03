import pandas as pd
from tqdm import tqdm
import pinecone
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
import streamlit as st
import openai


@st.experimental_singleton
def get_data():
    data = pd.read_csv("earnings_calls_sentencewise.csv")
    return data


# Initialize models from HuggingFace


@st.experimental_singleton
def get_t5_model():
    return pipeline("summarization", model="t5-small", tokenizer="t5-small")


@st.experimental_singleton
def get_flan_t5_model():
    return pipeline(
        "summarization", model="google/flan-t5-small", tokenizer="google/flan-t5-small"
    )


@st.experimental_singleton
def get_mpnet_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", device=device
    )
    model.max_seq_length = 512
    return model


@st.experimental_singleton
def get_sgpt_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        "Muennighoff/SGPT-125M-weightedmean-nli-bitfit", device=device
    )
    model.max_seq_length = 512
    return model


@st.experimental_memo
def save_key(api_key):
    return api_key


def query_pinecone(query, top_k, model, index, threshold=0.5):
    # generate embeddings for the query
    xq = model.encode([query]).tolist()
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    # filter the context passages based on the score threshold
    filtered_matches = []
    for match in xc["matches"]:
        if match["score"] >= threshold:
            filtered_matches.append(match)
    xc["matches"] = filtered_matches
    return xc


def format_query(query_results):
    # extract passage_text from Pinecone search result
    context = [result["metadata"]["Text"] for result in query_results["matches"]]
    return context


def sentence_id_combine(data, query_results, lag=2):
    # Extract sentence IDs from query results
    ids = [result["metadata"]["Sentence_id"] for result in query_results["matches"]]
    # Generate new IDs by adding a lag value to the original IDs
    new_ids = [id + i for id in ids for i in range(-lag, lag + 1)]
    # Remove duplicates and sort the new IDs
    new_ids = sorted(set(new_ids))
    # Create a list of lookup IDs by grouping the new IDs in groups of lag*2+1
    lookup_ids = [
        new_ids[i : i + (lag * 2 + 1)] for i in range(0, len(new_ids), lag * 2 + 1)
    ]
    # Create a list of context sentences by joining the sentences corresponding to the lookup IDs
    context_list = [
        ". ".join(data.Text.iloc[lookup_id].to_list()) for lookup_id in lookup_ids
    ]
    return context_list


def text_lookup(data, sentence_ids):
    context = ". ".join(data.iloc[sentence_ids].to_list())
    return context


def gpt3_summary(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=text + "\n\nTl;dr",
        temperature=0.1,
        max_tokens=512,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=1,
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
        stop=["\n"],
    )
    return response.choices[0].text


st.title("Abstractive Question Answering")

st.write("The app uses the quarterly earnings call transcripts for 10 companies (Apple, AMD, Amazon, Cisco, Google, Microsoft, Nvidia, ASML, Intel, Micron) for the years 2016 to 2020.")

query_text = st.text_input("Input Query", value="Who is the CEO of Apple?")

num_results = int(st.number_input("Number of Results to query", 1, 5, value=3))


# Choose encoder model

encoder_models_choice = ["SGPT", "MPNET"]

encoder_model = st.selectbox("Select Encoder Model", encoder_models_choice)


# Choose decoder model

decoder_models_choice = ["GPT3 (QA_davinci)", "GPT3 (summary_davinci)", "T5", "FLAN-T5"]

decoder_model = st.selectbox("Select Decoder Model", decoder_models_choice)


if encoder_model == "MPNET":
    # Connect to pinecone environment
    pinecone.init(
        api_key="ea9fd320-6f8a-4edd-bf41-9e972b95cbf9", environment="us-east1-gcp"
    )
    pinecone_index_name = "week2-all-mpnet-base"
    pinecone_index = pinecone.Index(pinecone_index_name)
    retriever_model = get_mpnet_embedding_model()

elif encoder_model == "SGPT":
    # Connect to pinecone environment
    pinecone.init(
        api_key="0d8215d7-4ad5-4c76-8c45-4a40c0f6a1b7", environment="us-east1-gcp"
    )
    pinecone_index_name = "week2-sgpt-125m"
    pinecone_index = pinecone.Index(pinecone_index_name)
    retriever_model = get_sgpt_embedding_model()


window = int(st.number_input("Sentence Window Size", 0, 3, value=0))

threshold = float(
    st.number_input(
        label="Similarity Score Threshold", step=0.05, format="%.2f", value=0.55
    )
)

data = get_data()

query_results = query_pinecone(
    query_text, num_results, retriever_model, pinecone_index, threshold
)

if threshold <= 0.60:
    context_list = sentence_id_combine(data, query_results, lag=window)
else:
    context_list = format_query(query_results)


st.subheader("Answer:")


if decoder_model == "GPT3 (summary_davinci)":
    openai_key = st.text_input(
        "Enter OpenAI key",
        value="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",
        type="password",
    )
    api_key = save_key(openai_key)
    openai.api_key = api_key
    output_text = []
    for context_text in context_list:
        output_text.append(gpt3_summary(context_text))
    generated_text = ". ".join(output_text)
    st.write(gpt3_summary(generated_text))

elif decoder_model == "GPT3 (QA_davinci)":
    openai_key = st.text_input(
        "Enter OpenAI key",
        value="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",
        type="password",
    )
    api_key = save_key(openai_key)
    openai.api_key = api_key
    output_text = []
    for context_text in context_list:
        output_text.append(gpt3_qa(query_text, context_text))
    generated_text = ". ".join(output_text)
    st.write(gpt3_qa(query_text, generated_text))

elif decoder_model == "T5":
    t5_pipeline = get_t5_model()
    output_text = []
    for context_text in context_list:
        output_text.append(t5_pipeline(context_text)[0]["summary_text"])
    generated_text = ". ".join(output_text)
    st.write(t5_pipeline(generated_text)[0]["summary_text"])

elif decoder_model == "FLAN-T5":
    flan_t5_pipeline = get_flan_t5_model()
    output_text = []
    for context_text in context_list:
        output_text.append(flan_t5_pipeline(context_text)[0]["summary_text"])
    generated_text = ". ".join(output_text)
    st.write(flan_t5_pipeline(generated_text)[0]["summary_text"])

show_retrieved_text = st.checkbox("Show Retrieved Text", value=False)

if show_retrieved_text:

    st.subheader("Retrieved Text:")

    for context_text in context_list:
        st.markdown(f"- {context_text}")
