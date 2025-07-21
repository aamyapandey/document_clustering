# scripts/topic_modeling.py

import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def run_topic_modeling(docs, num_topics):
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(docs)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    st.subheader("🧠 Top Words Per Topic")
    words = vectorizer.get_feature_names_out()
    for i, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-10:]]
        st.markdown(f"**Topic {i + 1}:** " + ", ".join(top_words))
