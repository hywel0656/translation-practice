import streamlit as st
import json
import random
from sentence_transformers import SentenceTransformer, util

# Load the model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load translation data
with open("data/translations.json", "r", encoding="utf-8") as f:
    translations = json.load(f)

# Session state setup
if "current_index" not in st.session_state:
    st.session_state.current_index = random.randint(0, len(translations) - 1)
if "show_answer" not in st.session_state:
    st.session_state.show_answer = False
if "score" not in st.session_state:
    st.session_state.score = None

# Current item
item = translations[st.session_state.current_index]
japanese = item["japanese"]
correct_english = item["english"]

# App UI
st.title("Japanese → English Translation Practice")
st.subheader("Translate the following Japanese sentence into English:")

st.markdown(f"**{japanese}**")

user_input = st.text_input("Your English translation:")

if st.button("Check"):
    st.session_state.show_answer = True
    # Compute cosine similarity
    emb_user = model.encode(user_input, convert_to_tensor=True)
    emb_correct = model.encode(correct_english, convert_to_tensor=True)
    similarity = util.cos_sim(emb_user, emb_correct).item()
    st.session_state.score = similarity

if st.session_state.show_answer:
    st.markdown("---")
    st.markdown(f"**Expected translation:** {correct_english}")
    if st.session_state.score is not None:
        score_pct = round(st.session_state.score * 100, 2)
        st.markdown(f"**Similarity score:** {score_pct}%")
        if score_pct > 90:
            st.success("Excellent! ✅")
        elif score_pct > 70:
            st.info("Pretty good! 👍")
        else:
            st.warning("Not very close. Try again! ❗")

if st.button("Next"):
    st.session_state.current_index = random.randint(0, len(translations) - 1)
    st.session_state.show_answer = False
    st.session_state.score = None
    st.rerun()
