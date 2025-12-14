import streamlit as st
import os
import time

from model_config.testing import load_model_tokenizer, predict_value

st.set_page_config(page_title='Movie Review Analysis', page_icon='🦾')

MODEL_PATH = './model/classifier.keras'
TOKENIZER_PATH = './model/tokenizer.pkl'

@st.cache_resource
def get_model_and_tokenizer():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None, None
    
    model, tokenizer = load_model_tokenizer()
    return model, tokenizer

st.title("Movie Review Analysis")
st.subheader("Powered by LSTM")


with st.spinner("Loading AI Models..."):
    model, tokenizer = get_model_and_tokenizer()

if model is None or tokenizer is None:
    st.error("Model or Tokenizer missing!")
    st.warning("Please run **model_config/training.py** to generate the model files.")
    st.stop()


input_val = st.text_area(
    label="Review Input", 
    placeholder="Type a movie review (e.g., 'The plot was amazing but the acting was stiff.')",
    height=150
)


if st.button('Analyze Sentiment', type="primary"):
    if not input_val.strip():
        st.warning("Please enter some text to analyze.", icon="⚠️")
        st.stop()

    raw_score = predict_value(text=input_val, model=model, tokenizer=tokenizer)[0][0]

    if raw_score > 0.5:
        label = "Positive"
        confidence = raw_score
    else:
        label = "Negative"
        confidence = 1 - raw_score 
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"## Sentiment: {label}")
    
    with col2:
        st.metric(label="Confidence Level", value=f"{confidence*100:.2f}%")
        
    st.caption(f"Sentiment Scale : {raw_score}")
    st.progress(float(raw_score))