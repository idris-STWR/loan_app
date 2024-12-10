import streamlit as st
from backend import preprocessing as pp

#-----------------------Frontend processing------------------------

#---------------------------Title---------------------------
st.title('Loan Predictor app')


#--------------------------Widget Features---------------------------
gender = st.selectbox( "Gender",['Male','Female'])
married = st.selectbox('Relationship Status',['Single','Married'])
Dependents = st.selectbox('Number of Dependents',["none",'1','2',"3+"])
Education = st.selectbox("Education Status",['Graduate','Not Graduate'])
Employment = st.selectbox("Employment Status",["True","False"])
applicant = st.number_input('Applicant Income',help="Enter your annual income in dollars",format='%.f')
co_applicant = st.number_input('Co-Applicant Income',help="Enter your partner's annual income in dollars",format='%.f')
loan_amount = st.number_input('Loan Amount',help="Enter requested loan amount",format='%.f')
loan_amount_term = st.number_input('Loan Amount Term',help="Enter loan term in days",format='%.f')
credit_history = st.selectbox('Credit History',['True','False'])
area = st.selectbox("Property Area",['Rural','Semi Urban','Urban'])


#---------------------widget features---------------------
features = [gender,married,Dependents,Education,Employment,applicant,co_applicant,loan_amount,loan_amount_term,
            credit_history,area]



#-----------------------Backend processing-------------------------------
dataset = pp.list_to_dataframe(features)
print(dataset)

encoding = pp.categorical_to_numeric(dataset)
print(encoding.info())

addition = pp.column_addition(encoding)
print(addition.columns)

addition = addition.fillna(1)

standard = pp.data_standardization(addition) 
print(standard)

result = pp.model_deserialization('predictor.pkl', standard)
print(result)

#----------------------------button-------------
if st.button('Predict Loan Eligibility'):
    
    if result == 1:
        st.success('Loan Approed')

    else:
        st.error('Loan Disapproved')