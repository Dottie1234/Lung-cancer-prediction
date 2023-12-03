import pandas as pd
import streamlit as st
import pickle
import matplotlib.pyplot as plt


st.title('LUNG CANCER PREDICTION')
model_lr = pickle.load(open('lr_model', 'rb'))

tab1, tab2 = st.tabs(['Single prediction', 'Multiple prediction'])

with tab1:
    st.header('DIAGNOSIS QUESTION')

    st.subheader('These are the questions that would the determine if you are not sure that you have lung cancer.')
    
    age = st.slider('How old are you? ', 20, 90, step= 1)
    gender = st.radio('Gender', ['Male', 'Female'])
    smoking = st.radio('Do you smoke? ', ['No', 'Yes'])
    yellow_fingers = st.radio('do you have yellow fingers? ', ['No', 'Yes'])
    anxiety = st.selectbox('do you have anxiety? ', ['No', 'Yes'])
    chest_pain = st.selectbox('do you have chest_pain? ', ['No', 'Yes'])
    chronic_disease = st.selectbox('do you have any chronic diseases? ', ['No', 'Yes'])
    fatigue = st.selectbox('Are you regularly tired? ', ['No', 'Yes'])
    allergy = st.selectbox('Do you have any allergy?', ['No', 'Yes'])
    wheezing = st.selectbox('Do you wheeze when you cough? ', ['No', 'Yes'])
    alcohol = st.selectbox('Do you take alcohol?', ['No', 'Yes'])
    coughing = st.selectbox('Are you coughing?', ['No', 'Yes'])
    breath = st.selectbox('Are you able to breath well? ', ['No', 'Yes'])
    swallow = st.selectbox('do you have any swallowing difficulties', ['No', 'Yes'])

    data = pd.DataFrame({'GENDER': [gender], 'AGE': [age], 'SMOKING': [smoking] , 'YELLOW_FINGERS': [yellow_fingers], 'ANXIETY': [anxiety], 'CHRONIC DISEASE': [chronic_disease], 'FATIGUE ': [fatigue], 'ALLERGY ': [allergy], 'WHEEZING': [wheezing], 'ALCOHOL CONSUMING': [alcohol], 'COUGHING': [coughing], 'SHORTNESS OF BREATH': [breath], 'SWALLOWING DIFFICULTY': [swallow], 'CHEST PAIN': [chest_pain]})

    st.write('')
    st.info('this is the overall answers you have choose down below, Please confirm what you have chosen')
    st.write(data)

    data['GENDER'] = data['GENDER'].apply(lambda x: 0 if x == 'Female' else 1)
    col = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING','ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH','SWALLOWING DIFFICULTY', 'CHEST PAIN']

    for feature in col:
        data[feature] = data[feature].apply(lambda x: 0 if x == 'No' else 1)
        
    

    if st.button('Submit'):
        pred = model_lr.predict(data)
        prob = model_lr.predict_proba(data)
        trans = prob.transpose()
        pred_proba = model_lr.predict_proba(data)[:,1]*100
        if pred == 0:
            st.balloons()
            st.write('You do not have lung cancer')
            st.write(f'your probability of having lung cancer is {pred_proba}')     
            plt.bar(trans[0], height=0.1)
        else:
            st.write('You are diagnosed to have lung cancer')
            st.write(f'your probability of having lung cancer is {pred_proba}')

with tab2:
    st.info('Upload the file to be predicted down below')
    file = st.file_uploader('Upload here', type = ['csv', 'txt', 'xlsx'])
    if file == None:
        st.write('You have not uploded any file')
    else:
        st.info('you uploaded the correct file format')
        st.write()
        df = pd.read_csv(file)
        st.write(df)
        if 'GENDER' in df.columns:
            df = df[['GENDER', 'AGE','SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']]

            df['GENDER'] = df['GENDER'].apply(lambda x: 0 if x == 'F' else 1)
            def encode(features):
                for feature in features:
                    df[feature] = df[feature].apply(lambda x: 0 if x == 'No' else 0 if x == 1 else 1 )
            
            encode(['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'])
            df
            
            pred = model_lr.predict(df)
            prob = model_lr.predict_proba(df)
            d = pd.DataFrame(pred, columns = ['CANCER PREDICTION'])
            
            joined = pd.concat([df, d], axis = 1)

            joined['CANCER PREDICTION'] = joined['CANCER PREDICTION'].apply(lambda x: 'True' if x == 1 else 'False')
            st.info('below is the prediction made by the web app')
            st.write('')
            st.write(joined)
            
            def convert_df(pred):
                return df.to_csv().encode('utf-8')

            csv = convert_df(joined)

            st.download_button(label="Download prediction as CSV",data=csv,mime='text/csv')
        else: 
            st.info('You have not submitted the file with the correct features for the prediction, Please upload the correct file for the accurate prediction')