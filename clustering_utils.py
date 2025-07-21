import streamlit as st
import pandas as pd
import os
import zipfile
import shutil
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Helper function to clean text
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)
    return text.lower()

# Unzips and reads files from uploaded folder
def extract_texts_from_zip(zip_file):
    temp_dir = "temp_upload"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    text_data = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".txt") or '.' not in file:  # assumes plain text
                try:
                    with open(os.path.join(root, file), 'r', encoding="latin1") as f:
                        content = f.read()
                        text_data.append(preprocess_text(content))
                except Exception as e:
                    st.warning(f"Failed to read file {file}: {e}")
    return text_data

# LDA visualization function
def display_lda_topics(model, feature_names, n_top_words):
    topics = []
    for idx, topic in enumerate(model.components_):
        words = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        topics.append(f"Topic {idx+1}: {words}")
    return topics

# Streamlit UI
st.set_page_config(page_title="Text Clustering & Topic Modeling", layout="wide")
st.title("Text Clustering & Topic Modeling App")

st.sidebar.header("Upload Dataset")
zip_file = st.sidebar.file_uploader("Upload a .zip folder of text files", type=["zip"])

if zip_file:
    documents = extract_texts_from_zip(zip_file)

    if documents:
        st.sidebar.subheader("Choose Algorithm")
        method = st.sidebar.radio("Select Clustering Method", ("KMeans", "LDA"))

        if method == "KMeans":
            num_clusters = st.sidebar.slider("Number of Clusters", 2, 20, 5)
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            X = vectorizer.fit_transform(documents)

            model = KMeans(n_clusters=num_clusters, random_state=42)
            model.fit(X)
            labels = model.labels_

            st.subheader("KMeans Clustering Results")
            for i in range(num_clusters):
                cluster_docs = [doc for doc, label in zip(documents, labels) if label == i]
                combined = " ".join(cluster_docs)
                wordcloud = WordCloud(width=800, height=400).generate(combined)
                st.markdown(f"#### Cluster {i + 1}")
                st.image(wordcloud.to_array(), use_column_width=True)

        elif method == "LDA":
            num_topics = st.sidebar.slider("Number of Topics", 2, 20, 5)
            num_words = st.sidebar.slider("Words per Topic", 5, 20, 10)

            count_vectorizer = CountVectorizer(stop_words='english', max_features=1000)
            doc_term_matrix = count_vectorizer.fit_transform(documents)

            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda.fit(doc_term_matrix)

            st.subheader("LDA Topic Modeling Results")
            topics = display_lda_topics(lda, count_vectorizer.get_feature_names_out(), num_words)
            for topic in topics:
                st.write(topic)
    else:
        st.error("No readable text files found in uploaded ZIP.")
else:
    st.info("Please upload a .zip file of text documents.")
