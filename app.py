import streamlit as st
import joblib
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
stop_words = set(stopwords.words('english'))

# Preprocessing Function
def preprocess(text):
    text = text.lower()
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

# Page Config
st.set_page_config(page_title="🎬 Movie Review Sentiment", layout="centered")
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🎥 Movie Review Sentiment Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Enter a review and find out if it's Positive or Negative 🎭</p>", unsafe_allow_html=True)

st.markdown("----")

# Text Input
review = st.text_area("📝 Write your movie review below:", height=200)

# Predict Button
if st.button("🔮 Predict Sentiment"):
    if not review.strip():
        st.warning("⚠️ Please enter a review first.")
    else:
        processed = preprocess(review)
        vector = vectorizer.transform([processed])
        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        if prediction == 1:
            st.success("✅ Sentiment: Positive 🎉")
            st.progress(int(prob[1] * 100))
        else:
            st.error("❌ Sentiment: Negative 😢")
            st.progress(int(prob[0] * 100))

st.markdown("----")
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
