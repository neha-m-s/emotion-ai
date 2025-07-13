import streamlit as st
from emotion_cnn import detect_emotion_from_frame
from emotion_nlp import analyze_text_emotion, generate_response
import cv2
from PIL import Image
import numpy as np

st.set_page_config(page_title="Emotion-Aware Mental Health Companion", layout="centered")

st.title("🧠 Emotion-Aware Mental Health Companion")
st.markdown("An AI-powered assistant that detects your emotions through **facial expressions** and **text input** to offer empathetic responses in real-time.")

# Mode Selector
mode = st.selectbox("👇 Choose how you'd like to interact:", ["📷 Facial Emotion", "💬 Text Chat"])

# ----------------------------
# MODE 1: Facial Emotion (CNN)
# ----------------------------
if mode == "📷 Facial Emotion":
    st.header("📷 Real-Time Facial Emotion Detection")

    run = st.checkbox("🎥 Turn on Camera")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        success, frame = camera.read()
        if not success:
            st.error("Unable to access camera.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        emotion = detect_emotion_from_frame(frame)

        # Show frame and emotion
        st.markdown(f"😊 **Detected Emotion**: `{emotion}`")
        FRAME_WINDOW.image(frame)

    if not run:
        st.info("Camera is off. Click the checkbox to activate.")

# ----------------------------
# MODE 2: Text-Based Chat (NLP)
# ----------------------------
elif mode == "💬 Text Chat":
    st.header("💬 Talk to your Emotion Companion")

    user_input = st.text_area("📝 What's on your mind?", height=150)

    if st.button("🔍 Analyze & Respond"):
        if user_input.strip() == "":
            st.warning("Please enter a message to analyze.")
        else:
            emotion = analyze_text_emotion(user_input)
            reply = generate_response(user_input, emotion)

            emoji_map = {
                "sadness": "😢", "joy": "😊", "anger": "😠", "fear": "😨",
                "gratitude": "🙏", "love": "❤️", "disappointment": "😞",
                "neutral": "😐", "nervousness": "😰", "surprise": "😮",
                "embarrassment": "😳", "amusement": "😄", "confusion": "😕"
            }

            emoji = emoji_map.get(emotion.lower(), "🧠")

            st.success(f"🧠 **Detected Emotion**: {emoji} `{emotion}`")
            st.markdown(f"🤖 **AI Response**: {reply}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ by Neha | Powered by CNN · Transformers · Streamlit")
