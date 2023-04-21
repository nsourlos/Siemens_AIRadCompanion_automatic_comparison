# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 18:57:52 2022

@author: 6000045970000731
"""
#Import dependencies
import pandas as pd
import os

ground_truth_path=os.getcwd()+"/Redcap_exports_BMI/" #Path of REDCap export(s)
save_path=os.getcwd()+"/pat_BMI/" #Path to save one csv file for each participant

#Get patient ids from files containing the scans
BMI_pats=[]
for BMI_pat in os.listdir(os.getcwd()+"\BMI_exp\BMI_high_scans"):
    BMI_pats.append(int(BMI_pat))
for BMI_pat in os.listdir(os.getcwd()+"\BMI_exp\BMI_low_scans"):
    BMI_pats.append(int(BMI_pat))


for file in os.listdir(ground_truth_path): #Loop over REDCap exports (in cases there are multiple with eg. different emphysema degrees)
    
    if not os.path.exists(save_path+file): #Create folder to save images
        os.mkdir(save_path+str(file))
    
    ground_truth_nodules=pd.read_csv(ground_truth_path+file) #Get one export file
    for patient in ground_truth_nodules['participant_id']: #Loop over participants of that export
        if patient in BMI_pats: #If that participant in the list of participants of our experiment
            ground_truth_nodules[ground_truth_nodules['participant_id']==patient].to_csv(save_path+file+'/'+str(patient)+'.csv',
                                                                                     index=False) #Save it to csv
