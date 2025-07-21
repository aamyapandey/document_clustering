import streamlit as st
import os
import zipfile
import tempfile
import shutil

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

import numpy as np


def read_txt_files_from_zip(zip_file):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    documents = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                    documents.append(f.read())
    shutil.rmtree(temp_dir)
    return documents


def read_txt_file(file):
    return [file.read().decode('utf-8')]


def vectorize_text(documents, method='tfidf'):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        # Handle small datasets
        min_df_val = 1
        max_df_val = 1.0
        if len(documents) > 3:
            min_df_val = 2
            max_df_val = 0.95
        vectorizer = CountVectorizer(stop_words='english', max_df=max_df_val, min_df=min_df_val)

    X = vectorizer.fit_transform(documents)
    return X, vectorizer


def perform_kmeans(X, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans


def perform_lda(X, num_topics):
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    return lda


def display_top_terms(model, vectorizer, num_terms, model_type='kmeans'):
    terms = vectorizer.get_feature_names_out()
    components = model.cluster_centers_ if model_type == 'kmeans' else model.components_
    for idx, topic in enumerate(components):
        st.write(f"**Topic {idx + 1}:**")
        top_terms = topic.argsort()[-num_terms:][::-1]
        st.write(", ".join([terms[i] for i in top_terms]))

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
            " ".join([terms[i] for i in top_terms]))
        st.image(wordcloud.to_array())


# --- Streamlit App ---
st.title("🧠 Topic Modeling & Clustering (KMeans + LDA)")

uploaded_file = st.file_uploader("Upload a .txt file or a .zip of .txt files", type=["zip", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/zip" or uploaded_file.name.endswith(".zip"):
        documents = read_txt_files_from_zip(uploaded_file)
    elif uploaded_file.type == "text/plain" or uploaded_file.name.endswith(".txt"):
        documents = read_txt_file(uploaded_file)
    else:
        st.error("Only .txt or .zip of .txt files are supported.")
        st.stop()

    if len(documents) == 0:
        st.warning("No valid text documents found.")
        st.stop()

    model_choice = st.sidebar.selectbox("Choose a model", ["KMeans", "LDA"])
    num_topics = st.sidebar.slider("Number of topics/clusters", 2, 10, 3)

    if model_choice == "KMeans":
        X, vectorizer = vectorize_text(documents, method='tfidf')
        kmeans = perform_kmeans(X, num_topics)
        display_top_terms(kmeans, vectorizer, 10, model_type='kmeans')

        # Show cluster distribution
        cluster_counts = np.bincount(kmeans.labels_)
        sns.barplot(x=list(range(num_topics)), y=cluster_counts)
        plt.xlabel("Cluster")
        plt.ylabel("Document Count")
        st.pyplot(plt)

    elif model_choice == "LDA":
        X, vectorizer = vectorize_text(documents, method='count')
        lda = perform_lda(X, num_topics)
        display_top_terms(lda, vectorizer, 10, model_type='lda')

        # Show document-topic distribution
        topic_distribution = lda.transform(X)
        avg_topic_dist = topic_distribution.mean(axis=0)

        sns.barplot(x=[f"Topic {i+1}" for i in range(num_topics)], y=avg_topic_dist)
        plt.xticks(rotation=45)
        plt.ylabel("Average Topic Distribution")
        st.pyplot(plt)

else:
    st.info("Please upload a `.txt` or `.zip` file containing `.txt` files.")

