from nltk.stem.porter import PorterStemmer
import time
from nltk.corpus import stopwords
import streamlit as st
import re
import string
import nltk
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Hide menu and footer
hide_menu = """
<style>
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

# Initialize Porter Stemmer
ps = PorterStemmer()

# Load saved model and vectorization configuration
model = tf.keras.models.load_model('./model_train.h5')
model.make_predict_function()

with open('./vectorization_config.pkl', 'rb') as f:
    tfidf = pickle.load(f)

loaded_vectorization_layer = tf.keras.layers.TextVectorization.from_config(tfidf['config'])
loaded_vectorization_layer.set_vocabulary(tfidf['vocabulary'])

# Streamlit app layout
st.title("Cyber-Bullying Detection🔍")
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)
input_text = st.text_area("**_Enter the text to analyze_**", key="**_Enter the text to analyze_**")

col1, col2 = st.columns([1, 6])

with col1:
    button_predict = st.button('Predict')

with col2:
    def clear_text():
        st.session_state["**_Enter the text to analyze_**"] = ""

    button_clear = st.button("Clear", on_click=clear_text)

st.markdown("---")

# Predict button animations
if button_predict:
    if input_text == "":
        st.snow()
        st.warning("Please provide some text!")
    else:
        with st.spinner("**_Prediction_** in progress. Please wait 🙏"):
            time.sleep(3)

        # Preprocess the input text
        cleanText = re.sub(r'http\S+|@[^\s]+[\s]?|#[^\s]+[\s]?|:[^\s]+[\s]?|[^ a-zA-Z0-9]|RT|[0-9]', '', input_text)
        transformText = " ".join([ps.stem(i) for i in nltk.word_tokenize(cleanText.lower()) if i.isalnum() and
                                  i not in stopwords.words('english') and i not in string.punctuation])

        # Vectorize the input text
        vectorized_input = loaded_vectorization_layer([input_text])

        # Predict the result
        result = np.argmax(model.predict(vectorized_input)[0])

        # Display the result
        if result == 1:
            st.subheader("Result")
            st.warning(":yellow[**Mất văn hóa**]")
        elif result == 2:
            st.subheader("Result")
            st.error(":red[**Tục**]")
        else:
            st.subheader("Result")
            st.success(":green[**Không tục**]")