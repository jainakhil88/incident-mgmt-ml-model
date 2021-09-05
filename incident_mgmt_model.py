# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 20:42:40 2021

@author: Akhil Jain
"""
import pandas as pd
import numpy as np
import sys
import joblib
import pickle
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import time
import os
import psutil
       
class Model(object):

    categorical_non_ordinal_features = ['incident_state', 'active', 'made_sla','contact_type','knowledge','u_priority_confirmation', 'notify']
    categorical_ordinal_features = [ 'impact', 'urgency', 'priority']
    categorical_features = categorical_non_ordinal_features+categorical_ordinal_features
    numerical_features = ['reassignment_count', 'reopen_count', 'sys_mod_count']
    temporal_features = ['opened_at','sys_updated_at', 'closed_at']
    mixed_features = ['number','caller_id', 'sys_created_by','opened_by','sys_updated_by', 'location',
                  'category', 'subcategory', 'u_symptom', 'assignment_group', 'assigned_to', 'closed_code', 'resolved_by']
    
    mixed_features_engineered_columns= ["caller_id_num", "sys_created_by_num","sys_updated_by_num",
                                    "opened_by_num", "location_num","category_num","subcategory_num",
                                    "u_symptom_num","assignment_group_num","assigned_to_num",
                                    "closed_code_num","resolved_by_num"]
    
    columns_to_be_dropped = ["caller_id", "number", "opened_by", "sys_created_by",
              "sys_updated_by","location","category","subcategory", "u_symptom","assignment_group",
              "assigned_to","closed_code","resolved_by", "incident_number",'opened_at','sys_updated_at', 'closed_at']
    
    temporal_number_features=["sys_updated_at_ms"]

    date_format='%d-%m-%Y %H:%M'
    
    float_data_type_features=['reassignment_count', 'reopen_count', 'sys_mod_count','sys_updated_at_ms']

    int_data_type_features=['impact','urgency','priority','caller_id_num','sys_updated_by_num','location_num', 
                            'category_num', 'subcategory_num', 'closed_code_num', 'resolved_by_num', 'incident_state_Awaiting Evidence',
                            'incident_state_Awaiting Problem','incident_state_Awaiting User Info',
                            'incident_state_Awaiting Vendor', 'incident_state_Closed','incident_state_New',
                            'incident_state_Resolved','active_True','made_sla_True', 'contact_type_Email','contact_type_IVR',
                            'contact_type_Phone', 'contact_type_Self service','knowledge_True','u_priority_confirmation_True',
                            'notify_Send Email','sys_created_by_num','u_symptom_num','opened_by_num', 'assignment_group_num','assigned_to_num']

    saved_model_directory="."+os.sep+"saved_models"+os.sep
    
    incident_state_simple_imputer=None
    
    non_ordinal_ohe=None
    
    label_encoder_impact = None
    label_encoder_urgency = None
    label_encoder_priority = None
    
    numerical_standard_scaler = None
    
    temporal_number_scaler = None
    
    assigned_to_num_rfc = None
    assignment_group_num_rfc = None
    opened_by_num_rfc = None
    sys_created_by_num_rfc = None
    u_symptom_num_rfc = None
    
    most_frequent_simple_imputer_for_zero_value = None
    
    lgbm_regressor = None
    
    def memory_usage_psutil(self):
        # return the memory usage in MB
        process = psutil.Process(os.getpid())
        mem = process.memory_info()[0] / float(1024 ** 2)
        return mem
    
    def load_pickle_to_obj(self, filename):
        obj = None
        with open(filename, 'rb') as f:
            obj=pickle.load(f)
        return obj
    
    def load_joblib_to_obj(self, filename):
        loaded_model = joblib.load(filename)
        return loaded_model
    
    def impute_particular_column_with_saved_model(self, df, column_to_be_imputed, imputer_model):
         df[column_to_be_imputed] = df[column_to_be_imputed].astype('int')
         impute_pred_for_df = imputer_model.predict(df.drop(columns=[column_to_be_imputed]))
         df[column_to_be_imputed] = impute_pred_for_df
         return df
    
    def __init__(self):
        
        # df=pd.read_csv('./data/incident_event_log.csv')
       
        # df=df.replace({-100: np.NAN})
        
        self.non_ordinal_ohe= pickle.load(open(self.saved_model_directory+"non_ordinal_ohe.pkl", 'rb'))       
        
        
        print("incident_state_simple_imputer exists"+str(os.path.exists(self.saved_model_directory+"incident_state_simple_imputer.pkl")))
        
        
        self.incident_state_simple_imputer = pickle.load(open(self.saved_model_directory+"incident_state_simple_imputer.pkl", 'rb'))
        
        print("incident_state_simple_imputer object="+str(self.incident_state_simple_imputer))
        
        # df["incident_state"] = self.incident_state_simple_imputer.fit_transform(df[["incident_state"]]).ravel()
        
        # df=None
        
        self.label_encoder_impact = self.load_pickle_to_obj(self.saved_model_directory+"label_encoder_impact.pkl")
        
        
        self.label_encoder_urgency = self.load_pickle_to_obj(self.saved_model_directory+"label_encoder_urgency.pkl")
        
        
        self.label_encoder_priority = self.load_pickle_to_obj(self.saved_model_directory+"label_encoder_priority.pkl")
        
        
        self.numerical_standard_scaler = self.load_pickle_to_obj(self.saved_model_directory+"numerical_standard_scaler.pkl")
        
        
        self.temporal_number_scaler = self.load_pickle_to_obj(self.saved_model_directory+"temporal_number_scaler.pkl")
        
        self.assigned_to_num_rfc = self.load_joblib_to_obj(self.saved_model_directory+"assigned_to_num_rfc.sav")      
        self.assignment_group_num_rfc = self.load_joblib_to_obj(self.saved_model_directory+"assignment_group_num_rfc.sav")
        self.opened_by_num_rfc = self.load_joblib_to_obj(self.saved_model_directory+"opened_by_num_rfc.sav")
        self.sys_created_by_num_rfc = self.load_joblib_to_obj(self.saved_model_directory+"sys_created_by_num_rfc.sav")
        self.u_symptom_num_rfc = self.load_joblib_to_obj(self.saved_model_directory+"u_symptom_num_rfc.sav")
        
        self.most_frequent_simple_imputer_for_zero_value = self.load_pickle_to_obj(self.saved_model_directory+"most_frequent_simple_imputer_for_zero_value.pkl")
        
        self.lgbm_regressor = self.load_joblib_to_obj(self.saved_model_directory+"lgbm_reg_model.sav")
        print("init called")
        
        
        

    def predict(self, df: pd.DataFrame):
        
        #removing unused columns        
        df = df.drop(['cmdb_ci','problem_id','rfc','vendor','caused_by', 'resolved_at', 'sys_created_at'], axis=1)    
        
        #replacing missing values ? with None
        df=df.replace({'?': None})
        df=df.replace({-100: np.NAN})
        
        #converting to string/object type, this is for case where -100 was only passed in incident state for single row and pandas treat it as float
        df['incident_state']=df['incident_state'].astype(object)
        
        #converting to date time format
        df['opened_at'] = pd.to_datetime(df.opened_at, format=self.date_format)
        df['sys_updated_at'] = pd.to_datetime(df.sys_updated_at, format=self.date_format)
        df['closed_at'] = pd.to_datetime(df.closed_at, format=self.date_format)

        #calculatig y / target variable
        df['time_taken_to_complete'] = (df['closed_at'] - df['opened_at'])/ pd.Timedelta(days=1)
        
        #impute missing incident state if its value is -100 using simple imputer
        df["incident_state"]=self.incident_state_simple_imputer.transform(df[["incident_state"]]).ravel()
        
        #remove the string prefix part from all the columns
        df["incident_number"] = df["number"].str.replace("INC", "")
        
        '''Capturing numerical part for caller_id feature, remove Caller word from the feature '''
        df["caller_id_num"] = df["caller_id"].str.replace("Caller", "")
        df[["caller_id_num"]] = df[["caller_id_num"]].fillna(0)
        
        df["sys_created_by_num"] = df["sys_created_by"].str.replace("Created by", "")
        df[["sys_created_by_num"]] = df[["sys_created_by_num"]].fillna(0)
        
        df["sys_updated_by_num"] = df["sys_updated_by"].str.replace("Updated by", "")
        df[["sys_updated_by_num"]] = df[["sys_updated_by_num"]].fillna(0)
        
        df["opened_by_num"] = df["opened_by"].str.replace("Opened by", "")
        df[["opened_by_num"]] = df[["opened_by_num"]].fillna(0)
        
        df["location_num"] = df["location"].str.replace("Location", "")
        df[["location_num"]] = df[["location_num"]].fillna(0)
        
        df["category_num"] = df["category"].str.replace("Category", "")
        df[["category_num"]] = df[["category_num"]].fillna(0)
        
        df["subcategory_num"] = df["subcategory"].str.replace("Subcategory", "")
        df[["subcategory_num"]] = df[["subcategory_num"]].fillna(0)
        
        df["u_symptom_num"] = df["u_symptom"].str.replace("Symptom", "")
        df[["u_symptom_num"]] = df[["u_symptom_num"]].fillna(0)
        
        df["assignment_group_num"] = df["assignment_group"].str.replace("Group", "")
        df[["assignment_group_num"]] = df[["assignment_group_num"]].fillna(0)
        
        df["assigned_to_num"] = df["assigned_to"].str.replace("Resolver", "")
        df[["assigned_to_num"]] = df[["assigned_to_num"]].fillna(0)
        
        df["closed_code_num"] = df["closed_code"].str.replace("code", "")
        df[["closed_code_num"]] = df[["closed_code_num"]].fillna(0)

        df["resolved_by_num"] = df["resolved_by"].str.replace("Resolved by", "")
        df[["resolved_by_num"]] = df[["resolved_by_num"]].fillna(0)


        temp=self.non_ordinal_ohe.transform(df[self.categorical_non_ordinal_features])
        one_hot_encoded_column_names = self.non_ordinal_ohe.get_feature_names(self.categorical_non_ordinal_features)
        one_hot_encoded_df =  pd.DataFrame(temp, columns= one_hot_encoded_column_names, index=df.index)
        df = pd.concat([df, one_hot_encoded_df], axis='columns')
        df = df.drop(columns=self.categorical_non_ordinal_features)
        
        df['impact'] = self.label_encoder_impact.transform(df['impact'])
        df['urgency'] = self.label_encoder_urgency.transform(df['urgency'])
        df['priority'] = self.label_encoder_priority.transform(df['priority'])
        
        df['sys_updated_at_ms'] = pd.to_datetime(df['sys_updated_at'], unit='ms').astype(np.int64)
        
        df[self.numerical_features] = self.numerical_standard_scaler.transform(df[self.numerical_features])
     
        df[[self.temporal_number_features]] = self.temporal_number_scaler.transform(df[self.temporal_number_features])
        
        df=df.drop(columns = self.columns_to_be_dropped)
        
        y =  df['time_taken_to_complete']
        df = df.drop(['time_taken_to_complete'], axis=1)
        

        # if any values have 0, that is there are missing, we need to impute them from trained model
        if((df["assigned_to_num"] == 0).any()):
            df=self.impute_particular_column_with_saved_model(df,"assigned_to_num",self.assigned_to_num_rfc)

        if((df["assignment_group_num"] == 0).any()):
            df=self.impute_particular_column_with_saved_model(df,"assignment_group_num",self.assignment_group_num_rfc)
            
        if((df["opened_by_num"] == 0).any()):
            df=self.impute_particular_column_with_saved_model(df,"opened_by_num",self.opened_by_num_rfc)
            
        if((df["sys_created_by_num"] == 0).any()):
            df=self.impute_particular_column_with_saved_model(df,"sys_created_by_num",self.sys_created_by_num_rfc)
            
        if((df["u_symptom_num"] == 0).any()):
            df=self.impute_particular_column_with_saved_model(df,"u_symptom_num",self.u_symptom_num_rfc)
            
        # if there are still any missing values, impute using simple imputer
        df = pd.DataFrame(data=self.most_frequent_simple_imputer_for_zero_value.transform(df), 
                                        index=df.index, columns = df.columns)
        
        for col in self.float_data_type_features:
            df[col]=df[col].astype("float64")
            
        for col in self.int_data_type_features:
            df[col]=df[col].astype("int64")
        
        y_pred = self.lgbm_regressor.predict(df)

        return "Prediction: "+str(y_pred)+" | Metric(RMSE): "+str(np.sqrt(mean_squared_error(y, y_pred)))

