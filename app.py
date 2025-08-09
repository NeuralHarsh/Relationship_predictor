import streamlit as st
import pickle
import pandas as pd

st.write("✅ App loaded successfully!")

st.title("💘 Heart Hack")
st.write("Type your line—LoVe or FriendZone, the algorithm never lies.")

# Try loading model and vectorizer
try:
    model = pickle.load(open("NewBgCmodel.pkl", "rb"))
    emoji_vectorizer = pickle.load(open("Vectorizer.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Error loading model or vectorizer: {e}")
    st.stop()

# Input text
user_input = st.text_input("Your message:")

# Predict button
if st.button("Predict"):
    if user_input:
        try:
            features = emoji_vectorizer.transform([user_input])
            prediction = model.predict(features)[0]
            label = "💘 LoVers" if prediction == 1 else "🤝 FriendZone"
            st.success(f"Prediction: {label}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
    else:
        st.warning("Please enter a message first.")
