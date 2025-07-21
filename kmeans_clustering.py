# scripts/kmeans_clustering.py

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def run_kmeans_clustering(docs, num_clusters):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)

    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(tfidf_matrix)

    st.subheader("📍 Document Cluster Assignments")
    labels = km.labels_
    for i in range(num_clusters):
        st.markdown(f"**Cluster {i + 1}**")
        examples = [docs[j] for j in range(len(docs)) if labels[j] == i][:3]
        for ex in examples:
            st.markdown("- " + ex[:300] + "..." if len(ex) > 300 else "- " + ex)
