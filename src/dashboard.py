import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import os

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📈",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════
def get_prediction(text):
    try:
        response = requests.post(f"{API_URL}/predict", json={"text": text})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

# ══════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════
st.title("📈 Financial Sentiment Analysis")
st.markdown("""
Bu dashboard, finansal haberler ve tweetler üzerindeki duyguyu (sentiment) 
analiz etmek için eğitilmiş bir **FinBERT** modelini kullanır.
""")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area(
        "Analiz edilecek finansal metni girin:",
        placeholder="Örn: The company reported strong quarterly earnings with a 20% increase in revenue...",
        height=150
    )
    
    analyze_button = st.button("Analiz Et", type="primary")

if analyze_button and user_input:
    with st.spinner("Model analiz ediyor..."):
        result = get_prediction(user_input)
        
        if result:
            st.success("Analiz Tamamlandı!")
            
            # Ana Sonuç
            sentiment = result["sentiment"].upper()
            confidence = result["confidence"]
            
            st.metric("Tahmin Edilen Duygu", sentiment, f"{confidence:.2%} Confidence")
            
            # Skorlar
            scores = result["scores"]
            df_scores = pd.DataFrame({
                "Sentiment": list(scores.keys()),
                "Score": list(scores.values())
            })
            
            with col2:
                st.subheader("Olasılık Dağılımı")
                fig = px.pie(
                    df_scores, 
                    values="Score", 
                    names="Sentiment",
                    color="Sentiment",
                    color_discrete_map={
                        "positive": "#00CC96",
                        "neutral": "#636EFA",
                        "negative": "#EF553B"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

            st.json(result)
elif analyze_button and not user_input:
    st.warning("Lütfen bir metin girin.")

# Sidebar - API Health
st.sidebar.header("Sistem Durumu")
try:
    health = requests.get(f"{API_URL}/health").json()
    st.sidebar.success("API: Bağlı ✅")
    st.sidebar.info(f"Cihaz: {health.get('device', 'unknown')}")
except:
    st.sidebar.error("API: Bağlantı Kesildi ❌")

st.sidebar.markdown("---")
st.sidebar.caption("v1.0.0 | Financial Sentiment API")
