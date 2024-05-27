import streamlit as st
import pandas as pd
import joblib






st.sidebar.title('Employee Churn Analysis')

st.title("Churn PAGE")
st.title(":loudspeaker: BOSS HEAR ME")



html_temp = """
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Churn Prediction Streamlit server</title>
    <link rel="stylesheet" href="style.css">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Audiowide">
</head>

<body>

<div style="background-color:yellow;padding:10px">
<h2 style="color:white;text-align:right;"><li><a href="https://www.linkedin.com/in/muhsin-tosun/", target=>Contact: Muhsin  (linkedin.com/in/muhsin-tosun/)</a></li></h2>
</div>

<div style="background-color:lightblue;padding:10px">
<h2 style="color:white;text-align:center;">Welcome Boss, I'm Hr. Asistance </h2>

</div>"""
st.markdown(html_temp,unsafe_allow_html=True)

st.sidebar.title('Configure Your Customer')
satisfaction_level = st.sidebar.number_input("satisfaction_level", min_value =0.09, max_value = 1.0, value=0.53, step= 0.03)
last_evaluation = st.sidebar.number_input("last_evaluation", min_value =0.36, max_value = 1.0, value=0.42,  step= 0.03)
number_project = st.sidebar.number_input("number_project", min_value =2.0, max_value = 7.0, value=2.0, step= 1.0)
average_montly_hours = st.sidebar.number_input("average_montly_hours", min_value =96.0, max_value = 310.0, value=120.0, step= 10.0)
time_spend_company = st.sidebar.number_input("time_spend_company", min_value =2.0, max_value = 10.0, value=3.0,  step= 0.5)


model_name=st.selectbox('Select your ML model',('Xgboost','RanFor','KNN','GradBoost'))
if model_name=='Xgboost':
    model=joblib.load(open("XGBClassifier.pkl","rb"))
    st.success('You selected {} model'.format(model_name))
    
elif model_name=='KNN':
      
    model=joblib.load(open("KNeighborsClassifier.pkl","rb"))
    st.success('You selected {} model'.format(model_name))
    
elif model_name=='GradBoost':
    model=joblib.load(open("GradientBoostingClassifier.pkl","rb"))
    st.success('You selected {} model'.format(model_name))
    
elif model_name=='RanFor':
    model=joblib.load(open("RandomForestClassifier.pkl","rb"))
    st.success('You selected {} model'.format(model_name))

my_dict={

    "satisfaction_level":satisfaction_level,
    "last_evaluation": last_evaluation,
    "number_project": number_project,
    "average_montly_hours":average_montly_hours,
    "time_spend_company":time_spend_company,
       
}

df = pd.DataFrame.from_dict([my_dict])

columns=joblib.load(open('my_columns.pkl','rb'))

df=pd.get_dummies(df).reindex(columns=columns, fill_value=0)

st.table(df)

predict = st.sidebar.button("P R E D I C T")

if predict:
    if model_name=='RanFor':
        scaler=joblib.load(open('my_scaler_knn.pkl','rb'))
        df=scaler.transform(df)
        prediction=model.predict(df)
    else:
        prediction=model.predict(df)
    
    st.success('The estimation of your model is {} :' .format(int(prediction[0])))
    
    if prediction == 0:
        st.markdown("<h2 style='text-align: center; color: green;'> will stay  .</h2>", unsafe_allow_html=True)
        st.image("BI.jpg",output_format="auto")
    elif prediction == 1:
        st.markdown("<h2 style='text-align: center; color: red;'> will not stay</h2>", unsafe_allow_html=True)
        from streamlit_player import st_player
        st_player("https://www.youtube.com/watch?v=Uh3yhBy0m6c")



# Embed a music from SoundCloud
#st_player("https://soundcloud.com/imaginedragons/demons")
