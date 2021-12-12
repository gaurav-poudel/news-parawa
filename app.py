from PIL.Image import new
import numpy as np
import streamlit as st
import pandas as pd
import joblib,os
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer



#from streamlit.config import on_config_parsed
#from ml import func

# yo notebook bata ho 
#news_vectorizer = open("tfidi_model",'rb')
#news_cv = joblib.load(news_vectorizer)

knn_model = open("knn_model1.pkl",'rb')
model_knn = joblib.load(knn_model)

rfc_model = open("rfc_model1.pkl",'rb')
model_rfc = joblib.load(rfc_model)



ngram_range = (1,2)
max_df = 10
min_df = 1

max_features = 300


# yo directly dako , not book bata laudai k hunchha hunchha 
tfidf = TfidfVectorizer(encoding='utf-8',
                       ngram_range= ngram_range,
                       stop_words=None,
                       lowercase=False,
                       max_df= 1,
                       min_df = 1,
                       max_features= max_features,
                       norm = 'l2',
                       sublinear_tf= True)


#def get_key(val,my_dict):
    #for key,value in my_dict.items():
        #if val == value:
            #return key




#st.title("News Classifer ")


def main():
    activity = ['Home','Predication',"NLP"]
    choice = st.sidebar.selectbox("Select Activity",activity)
    if choice == 'Predication':
        st.header("This is Predication section")
        st.info("News Predication with ML")
        news_text = st.text_area("Enter News here","type here")
        str1 = 'this is to classify'
        all_ml_model = ["KNN","RFC"]

        model_choice = st.selectbox('chose a model',all_ml_model)

        prediction_labels = {'business': 0,'tech': 4,'sport': 3,'health': 5,'politics': 2,'entertainment': 1}

        #classify_btn = st.button("classify")
        if st.button("classify", key=None, help=str1, on_click=None, args=None, kwargs=None):
            #st.write("Orginal Text :: \n" , news_text)
            features_train = tfidf.fit_transform([news_text]).toarray()
            st.write(features_train)
            
            if model_choice == 'KNN':
                prediction = model_knn.predict(features_train)
                if prediction == 1:
                    st.warning("News is categorizes as :: {}".format('entertainment'))
                elif prediction == 2:
                    st.warning("News is categorizes as :: {}".format('politics'))
                elif prediction == 3:
                    st.warning("News is categorizes as :: {}".format('sport'))
                elif prediction == 4:
                    st.warning("News is categorizes as :: {}".format('tech'))
                elif prediction == 5:
                    st.warning("News is categorizes as :: {}".format('health'))
                
                #st.write(prediction)
                #result = get_key(prediction,prediction_labels)
                #st.warning("News is categorizes as :: {}".format(result))
            else:
                prediction = model_rfc.predict(features_train)
                st.write(prediction)
                
                #result1 = get_key(prediction,prediction_labels)
                #st.warning("News is categorizes as :: {}".format(result1))
        

                
    elif choice == 'Home':
        st.header("News Classifer")

    elif choice == 'NLP':
        st.title("This is the NLP section")
main()
    
