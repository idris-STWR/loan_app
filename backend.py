import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
pd.set_option('future.no_silent_downcasting',True)

class preprocessing():

    def __init__(self):
        pass


    def list_to_dataframe(features_list):
        values = features_list
        keys = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome','CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
       'Property_Area']
        
        dictionary = {}

        for i in range(len(values)):
            dictionary[keys[i]] = values[i]

        df = pd.DataFrame(dictionary, index=[0])
        return df
    

    #Here make sure you use the same columns names in the dictionary above and the same options
    # (drop down menue) in the frontend
    def categorical_to_numeric(dataset):
        
        streamlit_input = {
    'Married': {'Married': 1, 'Single': 0},
    'Gender': {'Male': 1, 'Female': 0},
    'Dependents': {'None': 0, '1': 1, '2': 2, '3+': 3},
    'Self_Employed': {'True': 1, 'False': 0},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Credit_History': {'True':1, 'False':0}
        }
#To replace all the converted values in the above dataset
        for features, replacements in streamlit_input.items():
            dataset[features] = dataset[features].replace(replacements)

        dataset = dataset.apply(pd.to_numeric, errors='coerce')

        return dataset
    
#Add applicant income to co-applicant

    def column_addition(dataset):
        
        try:

            dataset['ApplicantIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
            dataset = dataset.drop('CoapplicantIncome',axis=1)
            
            return dataset
        except Exception as e:
            print(f'Error:{e}')
    
#Standardize the numerical data. There are a couple of things to take note of. Because we're using a borrowed
#library we have to ensure there are no null values in our data. W e have to avoid any significant errors
#Because it can crash the whole site

    def data_standardization(data):
        if data.isnull().values.any():
            raise ValueError('Data Contains Missing Values')

        if not np.issubdtype(data.values.dtype, np.number):
            raise ValueError('Wrong data type')

        scaler = StandardScaler()
        result = scaler.fit_transform(data.values.reshape(-1,1))
        return result 

## Now we want to deserialize our model basically bring it in as a pickle file

    def model_deserialization(model, dataset):

        checker = dataset.reshape(1,-1)

        with open(model, 'rb') as f:
            predictor = pickle.load(f)

        result = predictor.predict(checker)
        return result

#-----------------------Steps for preprocessing-----------------
# 1. Convert our frontend data from a list format to a pandas dataframe
# 2. encode categorical data
# 3. Add applicant and co-applicat income
# 4. Standardize our features
# 5. deserialize our model and run the dataset through it