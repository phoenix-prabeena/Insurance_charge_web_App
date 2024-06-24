# ! pip install streamlit
import streamlit as st
import pickle
import numpy as np

lr1 = pickle.load(open('/home/misfit/Downloads/Python course/Projects/lr1.pkl','rb'))
dtr1 = pickle.load(open('/home/misfit/Downloads/Python course/Projects/dtr1.pkl','rb'))
rf1 = pickle.load(open('/home/misfit/Downloads/Python course/Projects/rfe1.pkl','rb'))




st.title('Insurance Charge Prediction Application')

st.header('fill the details to generate the predicted insurence charge')

#SelectBox
options = st.sidebar.selectbox('select ML Models',['Linear_regression','Decission_Tree','Random_Forest'])


# form widget
# inputbox, slidebar, drop down

Age = st.slider('Age',18,64)
Sex = st.selectbox('Sex',['Male','Female'])
BMI = st.slider('BMI',15,53)
Children = st.selectbox('Children',[0,1,2,3,4,5])
Smoker = st.selectbox('Smoker',['Yes','No'])
Region = st.selectbox('Region',['SEast','SWest','NEast','NWest'])


# 1 324 northeast
# 2 364 southeast
# 3 325 southwest
# 0 324 northwesr

if st.button('Predict'):

    if Sex =='Male':
        Sex = 1
    else :
        Sex = 0

    if Smoker == 'yes':
        Smoker = 1
    else:
        Smoker = 0

    if Region == "NWest":
        Region = 1
    elif Region == "NEast":
        Region = 0
    elif Region == "SEast":
        Region = 2
    else :
        Region = 3

    test = np.array([Age,Sex,BMI,Smoker,Children,Region])
    test = test.reshape(1,6)            #1 row and 6 columns
    if options == "Linear_regression,'Random_Forest":
        st.success(lr1.predict(test)[0])
    elif options == "Decission_Tree":
        st.success(dtr1.predict(test)[0])
    else:
        st.success(rf1.predict(test)[0])

    

