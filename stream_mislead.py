import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

#load model
model_misleading = pickle.load(open('model_misleading.sav', 'rb'))

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))

#judul title
st.title ('Deteksi Misleading Information')

clean_teks = st.text_input('Masukan Teks')

misleading_detection = ''

if st.button('Hasil Deteksi'):
    predict_misleading = model_misleading.predict(loaded_vec.fit_transform([clean_teks]))
    
    if (predict_misleading == 0):
        misleading_detection = 'mengandung misleading'
    else:
        misleading_detection = 'tidak mengandung misleading'

st.success(misleading_detection)