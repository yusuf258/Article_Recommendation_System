import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import random
import joblib

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Dinamik Makale Öneri Sistemi",
    page_icon="📚",
    layout="wide"
)

# --- VERİ VE MODEL FONKSİYONLARI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_raw_data():
    """CSV dosyasını dinamik olarak yükler."""
    file_path = os.path.join(BASE_DIR, 'articles.csv')
    if not os.path.exists(file_path):
        file_path = os.path.join(BASE_DIR, 'data.csv')
    
    if not os.path.exists(file_path):
        return None
        
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='latin1')
    
    # Sütun isimlerini temizle
    df.columns = df.columns.str.strip()
    return df

@st.cache_resource
def get_recommendations_matrix(df):
    """TF-IDF ve Cosine Similarity hesaplamalarını çalışma anında yapar veya modellerden yükler."""
    sim_matrix_path = os.path.join(BASE_DIR, 'models', 'similarity_matrix.pkl')
    
    if os.path.exists(sim_matrix_path):
        try:
            return joblib.load(sim_matrix_path)
        except:
            pass
            
    articles = df["Article"].tolist()
    tfidf = text.TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(articles)
    sim_matrix = cosine_similarity(matrix)
    return sim_matrix

# --- UYGULAMA BAŞLANGICI ---
st.title("📚 Akıllı Makale Öneri Sistemi")
st.markdown("Veri dosyasındaki makaleler analiz edilerek anlık öneriler üretilmektedir.")
st.markdown("---")

df = load_raw_data()

if df is not None:
    # Benzerlik matrisini dinamik hesapla
    sim_matrix = get_recommendations_matrix(df)
    titles = df["Title"].tolist()

    # Sidebar Kontrolleri
    st.sidebar.header("🔍 Makale Keşfi")
    
    if 'selected_index' not in st.session_state:
        st.session_state.selected_index = 0

    def handle_selection():
        st.session_state.selected_index = titles.index(st.session_state.temp_select)

    selected_title = st.sidebar.selectbox(
        "Bir makale seçin:", 
        titles, 
        index=st.session_state.selected_index,
        key="temp_select",
        on_change=handle_selection
    )

    if st.sidebar.button("🎲 Rastgele Öneri Getir"):
        st.session_state.selected_index = random.randint(0, len(titles) - 1)
        st.rerun()

    # Öneri ve İçerik Gösterimi
    if selected_title:
        idx = df[df["Title"] == selected_title].index[0]
        article_content = df.iloc[idx]["Article"]
        
        # Benzerlik skorlarını al ve en yakın 4 taneyi bul
        similarity_scores = sim_matrix[idx]
        # argsort()[-5:-1] -> Kendisi dışındaki en benzer 4 makale
        similar_indices = similarity_scores.argsort()[-5:-1]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📖 Mevcut Makale")
            st.info(f"### {selected_title}")
            st.write(article_content)
            
        with col2:
            st.subheader("🚀 Sizin İçin Önerilenler")
            st.caption("Bu içerikle en yüksek benzerliğe sahip makaleler:")
            
            # Önerileri tersten göster (en benzer en üstte)
            for i in reversed(similar_indices):
                rec_title = df.iloc[i]["Title"]
                with st.container(border=True):
                    st.success(f"**{rec_title}**")
                    if st.button(f"Makaleye Git ➔", key=f"btn_{i}"):
                        st.session_state.selected_index = i
                        st.rerun()
                
        st.markdown("---")
        with st.expander("📊 Kaynak Veri Setini İncele"):
            st.dataframe(df)
else:
    st.error("Makale veri dosyası (articles.csv veya data.csv) bulunamadı.")
