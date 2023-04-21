# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:56:31 2021

@author: soyrl
"""

#Import libraries and dependencies
import os
import pydicom as dicom
import numpy as np
import cv2
import time
# from termcolor import colored
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import sys # Save terminal output to txt file
import traceback #Print errors when occur
import torch
import torchvision.ops.boxes as bops #calculate overlap between two nodule boxes

#Input and Output paths
data_path= "H:\My Desktop\emphysema_all\emphysema_exp_scans/advanced_scans/" #Folders with scans
#For BMI "H:\My Desktop/BMI_exp/BMI_low_scans_new/"

output_path="H:/My Desktop/adv_29-3/" #Any name

#Path of ground truth nodules for a specific emphysema degree
ground_truth_path="H:\My Desktop\emphysema_all\emphysema_exp_scans/advanced_gt_only_new_added/" 
#For BMI "H:\My Desktop/pat_BMI_red_latest_new/"

#Path of AI nodules and their volume - created with file aorta_calcium_lungnodules.ipynb
AI_path= "H:\My Desktop/emphysema_all/allemphexper_AI_new_latest_14-3.xlsx"
#"H:\My Desktop/BMI_exp_AI_13-1.xlsx"

if not os.path.exists(output_path): #Create folder to save images
    os.mkdir(output_path)
    
#Create lists of input and output paths to use with Parallel processing library    
inputs=[]
outputs=[]

#Loop over all patient folders and add paths to input and output lists
for path,subdir,files in os.walk(data_path): #each time gets into next subdir and repeats
#path,subdir,files for a specific directory each time
    result_path='' #Initialize string to be filled with directory name
    if len(files)>0:
        result_path=output_path+path.split('/')[-1] #Path of specific patient
        
        if not os.path.exists(result_path): #Create folder to save images
            os.mkdir(result_path)
        
        outputs.append(result_path)
        inputs.append(path)

start=time.time() #To count time to run all steps

def CT_files_extract(input_path,output_path,dpi=1200):
    'Extracts all CT files (AI, original scans along with annotated CT images), the num of AI nodules,'
    'as well as the total slices in the SEG files, in AI output, and the total num of CT files (original scan).'
    'We also get possible errors. It is assumed that the input path contains only DICOM files of a specific patient'
    
    "input_path: Path of patient with nodules - should contain files of one patient only"
    "output_path: Path where information and images will be saved" 
    "dpi: Specifies resolution of output images"

    start=time.time() #To count the time for this function to run
    
    #Save output of this step to txt file
    sys.stdout = open(output_path+'_step1.txt', 'w') 

    if not os.path.exists(output_path): #Create folder to save images
        os.mkdir(output_path)
            
    num=0 #Counter of the number of files for this patient
    size_AI=0 #Count the number of AI slices - In case that SEG files have different number of slices
    size_CTs=0 #Count the number of CT files - original scan and annotations files together
    AI_slices=[] #To save number of AI slices that contain nodules

    #Initialize empty lists to save CT file names with images, and number of slices in SEG files below
    CTfiles=[]
    size_SEG=[]
    
    #Initialize empty dictionary to keep track of AI output slices and the number of nodules they contain
    AI_num_nods={}
    
    #List to be filled with problematic SEG files, if any, as well as unknown file types and info if SEG files do not have the same number of slices
    errors_SEG=[]

    #https://python.plainenglish.io/how-to-design-python-functions-with-multiprocessing-6d97b6db0214
    #https://github.com/tqdm/tqdm#usage
    for file in tqdm(os.listdir(input_path),desc='step1',ascii=True): #Parallel loop into each patient
        if input_path.endswith('/'): #Ensure that path ends with '/'
            file_path=input_path+file #subdir with files
        else:
            file_path=input_path+'/'+file 
        
        if os.path.isfile(file_path): #If we have a file and not a subdir
            dicom_file=dicom.dcmread(file_path) #Read file
            num+=1 #Increase files counter by 1
            print("Processing file {}/{} ...".format(num,len(os.listdir(input_path))))
         
            if dicom_file.Modality=='CT': #For CT images
                    image=dicom_file.pixel_array #Load CT slice
                    
                    if len(np.unique(image))==1: #Problem if only 0 values
                        print("CT image slice of file {} is empty".format(file))
                    
                    # Save CT slices based on their resolution
                    # Higher resolution images also have annotations on the figure                 
                    print("CT file name is: {}".format(file))
                    
                    # plt.ioff()
                    # plt.figure()
                    # plt.imshow(image) 
                    # plt.title(file)
                    # plt.savefig(output_path+'/'+str(image.shape[0])+'_'+str(file[:-4])+'.png',dpi=dpi) # file[:-4] was used to avoid '.dcm' ending
                    # plt.close()
                    #cv2 gives empty image since range from 0- ~3000 for original scan images
                            
                    if image.shape[0]==512: #Original scan slice or annotated image
                        CTfiles.append(file) #Save information about file name   
                        if len(file.split('.')[3])==1: #Increase size_CT only for original scan - just one value in that position eg. 4 or 6
                            size_CTs=size_CTs+1
                        
                    else:
                        if image.shape[0]==1024:
                            if np.sum(image[900:,960:])!=0: #Ensure that we don't have graphs with HU information

                                if file.split('.')[4] in AI_slices:
                                    continue
                                else:
                                    size_AI=size_AI+1 #Increase number of AI slices (should be same as original scan)
                                    AI_slices.append(file)#.split('.')[4])
                                
                                #Assess differences between channels to find red and yellow nodule annotations of AI
                                #Threshold if no nodules detected is ~500 (from 492-527 for patient 695269) - not from 'Not for clinical use' but from some black pixels!
                                
                                #If we plot image[np.where(image[:,:,1]!=image[:,:,2])] we will see a black vertical line of those different pixels
                                # if len(np.where(image[:,:,1]!=image[:,:,2])[0])<=600:
                                #     print("no AI for file {}".format(file))
                                #     print(len(np.where(image[:,:,1]!=image[:,:,2])[0]))
                                    
                                if len(np.where(image[:,:,1]!=image[:,:,2])[0])>600: #threshold for at least 1 nodule - 1000 for at least 2, if needed      
                                  
                                    #Resize AI image to (512,512) - same size as SEG and CT files below, convert to HSV and get mask for red and yellow
                                    AI_image=image.copy() #As a good practice - to ensure that we don't change the original image
                                    AI_512=cv2.resize(AI_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC) #Resize to (512,512)
                                    AI_hsv=cv2.cvtColor(AI_512,cv2.COLOR_BGR2HSV) #Convert from BGR to HSV colorspace
                                    mask_im_red=cv2.bitwise_and(AI_hsv,AI_hsv, mask=cv2.inRange(AI_hsv, (100,0,0), (200, 255, 255))) #Red mask - lower and upper HSV values
                                    mask_im_red[0:50,:,:]=0 #Set top pixels that mention 'Not for clinical use' to zero - ignore them and keep only nodules
                                    
                                    # Maybe needed for nodule identification - avoid duplicates - also (80,0,70) and (80,0,130) but best (80,100,0)
                                    # mask_im_yellow=cv2.bitwise_and(AI_hsv,AI_hsv, mask=cv2.inRange(AI_hsv, (80,0,110), (110, 255, 255))) #Yellow mask
                                    # cv2.imwrite(output_path+'/'+str(image.shape[0])+'_'+str(file[:-4])+'_yellow'+'.png',mask_im_yellow) #Save yellow mask - range from 0-255

                                    #Until here mask_im_red is a color image with range of values from 0-255
                                    #Now we convert from BGR (emerged with bitwise operation) to grayscale to have same shape (512,512) as SEG files
                                    #It is used below to take a thresholded image
                                    mask_im_red_gray=cv2.cvtColor(mask_im_red,cv2.COLOR_BGR2GRAY) #Also changes range of values
                                                                                   
                                    if len(np.unique(mask_im_red_gray))!=1: #If there is a nodule in the image

                                        #Get a smoothed (thresholded) image with only nodules
                                        mask_red_thresh = cv2.threshold(mask_im_red_gray,128,255,cv2.THRESH_BINARY)[1] 
                                        #Activate below to get mask without red box around it
                                        # cv2.imwrite(output_path+'/'+str(image.shape[0])+'_'+str(file[:-4])+'_mask_no_box'+'.png',mask_red_thresh) #Save it
                                        
                                        #Get contours for each nodule and create a rectangle around them
                                        # mask_im_red_RGB=cv2.cvtColor(mask_im_red_gray,cv2.COLOR_GRAY2RGB) #Convert grayscale to RGB (not BGR as before)
                                        contours = cv2.findContours(mask_red_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        contours = contours[0] if len(contours) == 2 else contours[1]
                                        if len(contours)>0:
                                            checks={} #Empty dictionary to keep track of nodule locations of a specific slice
                                            for index,cntr in enumerate(contours):
                                                x,y,w,h = cv2.boundingRect(cntr)
                                                checks[index]=[x,y,w,h]
                                            
                                            length=0 #find overlapping boxes
                                            for index,values in enumerate(list(checks.values())):
                                                x,y,w,h=values
                                                try: #To avoid run out of index error
                                                    for index1 in range(index+1,len(list(checks.values()))): #Loop over box locations and check for overlap
                                                        x1,y1,w1,h1=list(checks.values())[index1]
                                                        if (x<=x1 and x1<=x+w and y<=y1 and y1<=y+h):
                                                            length=length+1
                                                except:
                                                    pass
                                            
                                            if length>0:
                                                print("There are overlapping boxes in file {}".format(file))
                                                
                                            AI_num_nods[file]=len(contours)-length
                                            
                                        # for cntr in contours:
                                        #     x,y,w,h = cv2.boundingRect(cntr)
                                        #     cv2.rectangle(mask_im_red_RGB, (x, y), (x+w, y+h), (0, 0, 255), 1) #Last argument thickness of box
                                        #     cv2.imwrite(output_path+'/'+str(image.shape[0])+'_'+str(file[:-4])+'_thresh_box'+'.png', mask_im_red_RGB) #Plot boxes on colored image
                                             
                                            
                                        # #For a colored-like version
                                        # plt.ioff()
                                        # plt.figure()
                                        # plt.imshow(mask_im_red_gray) 
                                        # plt.title(file)
                                        # plt.savefig(output_path+'/'+str(image.shape[0])+'_'+str(file[:-4])+'_colored_thresh'+'.png',dpi=dpi) 
                                        # plt.close()
                                    
                    print('\n')
                    
            elif dicom_file.Modality=='SEG': #For SEG files  
                image=dicom_file.pixel_array #Load 3D segmentation file

                if np.sum(image)<5000:#Count number of proper segmentation files - If above that then we have yellow lines on bottom of the scan
                    print("SEG file {} has sum of pixels <5000 and equal to {}".format(file,np.sum(image)))
                    size_SEG.append(image.shape[0]) #Confirm below that all SEG files have the same number of slices
                    
                else:
                    print("File {} has sum of pixels {}".format(file,np.sum(image)))
                    errors_SEG.append('The following has more than 5000 pixels as Segmentation - empty curve line')
                    errors_SEG.append(file)
                    
                    #Just an empty curved line - Can be plotted below                 
                    # for curved_line in range(image.shape[0]):
                    #     plt.figure()
                    #     plt.imshow(image[curved_line,:,:]) 
                    #     plt.title(file)
                    #     plt.savefig(output_path+'/'+'512_lines_'+str(curved_line)+str(file[:-4])+'.png',dpi=dpi)
                    #     plt.close()
                   
                    # #or better with cv2
                    #     cv2.imwrite(output_path+'/'+'512_lines_'+str(curved_line)+str(file[:-4])+'.png',image[curved_line,:,:])
                        
            elif dicom_file.Modality=='SR': #For SR files - Will not be encountered
                    print("File {} is a SR".format(file)) 
                    print("\n")   
                                                
            else: #For any other type of file - We don't expect anything else
                print("File {} does not belong to any of the above categories and is a {} file".format(file, dicom_file.Modality))
                # print(dicom_file_final)
                print("\n")
                errors_SEG.append('Unexpected type of file occured in file:')
                errors_SEG.append(file)

    #Verify that all SEG files have same number of slices
    if len(np.unique(size_SEG))==1: #We should only have one number, the number of slices
        size_SEG=int(size_SEG[0]) #Convert list to number
    elif len(np.unique(size_SEG))==0: 
        errors_SEG.append('No Segmentation Files available') 
    else:
        print("ERROR: Segmentation files have different number of slices!")
        errors_SEG.append('Not all SEG files have the same number of slices for file'+str(input_path))
        errors_SEG.append('Here are the different num of slices for the previous file'+str(np.unique(size_SEG)))
 
    end=time.time()
    print("Total time to run CT extraction (step1) was {} secs".format(end-start))
    print("\n")  
    
    #Save output to text file
    sys.stdout.close()
    sys.stdout = sys.__stdout__   

    return CTfiles, size_SEG, errors_SEG, AI_num_nods, size_CTs, size_AI, AI_slices


def annotated_CTs_to_normal_scan(input_path,CT_files,output_path): 
    'Correlates annotated CT slices with slices of the original scan.'
    'It also returns empty CT slices and files with possible missed nodules due to low threshold'
     
    "input_path: Path of patient with nodules"
    "CT_files: Names of all CT files in a folder (output of step1 above)" 
        
    start=time.time() #To count the time for this function to run

    sys.stdout = open(output_path+'_step2.txt', 'w') #Save output to txt file

    original_CTs=[] #Empty list to be filled with names of original CT slices
    annotated_CT_files=[] #Empty list to be filled with names of annotated CT slices
    empty_CT_files=[] #List to be filled with files with errors
    possible_nodules_excluded=[] #To be filled with possible cases of nodules not found due to low threshold
    
    file_num=[x.split('.')[3] for x in CT_files] #Get numbers of CT files representing either original slice (eg. 4) or annotated slice (eg. 1042)
    annot_files_total=len([name for name in file_num if len(name)>1]) #This is the number of total annotation files available for a patient
    
    for index,annotated_CT in enumerate(tqdm(CT_files,desc='step2',ascii=True)): #Loop over CT_files to get annotated CT images
        annotated_CT_path=input_path+'/'+annotated_CT #Path of annotated CT file
        dicom_annotated_CT=dicom.dcmread(annotated_CT_path) #Read DICOM file
        image_annotated_CT=dicom_annotated_CT.pixel_array #Load CT slice
        # print(colored('Working on CT image number {}', 'blue').format(index)) #To print color output in terminal
        print('Working on CT image number {}'.format(index))

        #Check if above corresponds to annotated CT slice - Such files have a note of the bottom right corner of image ("F" symbol)
        if len(np.where(image_annotated_CT[410:,468:]!=0)[0])>100: #If there is an 'F' symbol then we have an annotated CT slice
            #Loop over all CT slices to find correspondance with annotated image
            for file in os.listdir(input_path):
                file_path=input_path+'/'+file #specific file path
                
                if os.path.isfile(file_path): #If we have a file and not a subdir
                    dicom_file=dicom.dcmread(file_path) #Read file
                 
                    if dicom_file.Modality=='CT': #Confirm that we indeed have CT slices here
                                image_CT=dicom_file.pixel_array #Load CT slice
                                
                                if len(np.unique(image_CT))==1: #Problem if only 0 values - should not happen
                                    print("CT image slice of file {} is empty".format(file))
                                    empty_CT_files.append(file)
                                    continue #to go to next file - or 'break' to exit loop
                               
                                # Find differences only in 512*512 CT slices - 1024*1024 are AI outputs
                                if image_CT.shape[0]==image_annotated_CT.shape[0] and len(file.split('.')[3])<3: #To get actual scan and not AI output
                                    
                                    #Find number of different pixel values between the 2 images                                         
                                    differences=np.where(image_annotated_CT!=image_CT)                                    
                    
                                    #To check if there are potential nodules missed by the low threshold
                                    if len(differences[0])>=1000 and len(differences[0])<60000:
                                        print("Number of different pixels are {} for files {} and {}".format(len(differences[0]),file,annotated_CT))
                                        possible_nodules_excluded.append([annotated_CT,file])
                                        
                                    #If the different pixels are not 0 (same as annotated CT image) and if it is less than 1000 pixels (threshold set by me)
                                    if len(differences[0])<1000 and len(differences[0]!=1):
                                        original_CTs.append(file)
                                        annotated_CT_files.append(annotated_CT)
                                        print("Annotated CT slice of file {} is the same as of CT file {}".format(annotated_CT,file))
                                        print("\n")  
                                        
                                        break #To save time otherwise keep looping - Stops 'correspondance' loop

                                else:    
                                    continue #Go to next file
        
        if len(annotated_CT_files)<annot_files_total: #When we looped in all available annotation CT files then break to avoid extra looping
            continue
        else:
            break
        
    end=time.time()
    print("Total time to run annotated_CT_to_normal_slices (step2) was {} secs".format(end-start))
    print("\n")  
    
    sys.stdout.close()
    sys.stdout = sys.__stdout__ 
                    
    return original_CTs, annotated_CT_files, empty_CT_files, possible_nodules_excluded 


def mask_seg_slices(original_CTs,input_path,output_path,annotated_CT_files,dpi=1200): 
    'Plots the nodules in the SEG files and returns correspondance between SEG files with original CT slices'
    'and annotated ones, as well SEG slices with errors. It also returns images with and without bounding boxes around nodules'
    'At last, it returns'
    
    "original_CTs: Original CT slices with nodules (output of step2 above)"
    "input_path: Path of patient with nodules"
    "output_path: Path where information and images will be saved"
    "annotated_CT_files: Annotated CT slices (output of step2 above)"
    "dpi: Specifies resolution of output images"
    
    start=time.time() #To count the time for this function to run
    
    #Save output to txt file
    sys.stdout = open(output_path+'_step3.txt', 'w') 
        
    #Empty lists to be filled with the final names of the files
    SEG_masks=[]
    original_CTs_final=[]
    annotated_CTs_final=[]
    SEG_masks_errors=[]
                
    for file in tqdm(os.listdir(input_path),desc='step3',ascii=True):
        file_path=input_path+'/'+file #specific file path
        
        if os.path.isfile(file_path): #If we have a file and not a subdir
            dicom_SEG=dicom.dcmread(file_path) #Read file

            if dicom_SEG.Modality=='SEG': #Only interested in SEG files  
                SEG_image_3D=dicom_SEG.pixel_array #Load 3D segmentation file
                              
                if np.sum(SEG_image_3D)<5000: #Some SEG files with sum>5000 give yellow lines - avoid that error
                    print("SEG file {} is being processed. The sum of SEG pixels is {}".format(file,np.sum(SEG_image_3D)))
                    print("\n")
                                                                                                             
                    SEG_3D=np.where(SEG_image_3D!=0) #Here are the pixels that belong to nodules   
                    
                    #Get slices in which nodules exist
                    nodule_slices=np.unique(SEG_3D[0])
                    print("{} are slices with nodules".format(nodule_slices))
                                           
                    #Get slices with maximum number of pixels having a nodule
                    nodule_sum=[0] #Initialize list of maximum numbers of pixels with 0 to keep track of that max
                    slicenum=[] #List to be filled with slices of max pixels
                    for slice in nodule_slices: #Loop over slices with nodules
                        if np.sum(SEG_image_3D[slice,:,:])>=nodule_sum[-1]: #If the maximum number of pixels in that slice is bigger or equal to the last max
                            nodule_sum.append(np.sum(SEG_image_3D[slice,:,:])) #Add that maximum to our list
                            slicenum.append(slice) #Add that slice to our list
                     
                    nodule_sum=np.array(nodule_sum) #Convert list to array
                    max_sum=np.max(nodule_sum) #Get maximum of all slices
                    index_max_sum=np.where(nodule_sum==max_sum) #Get index of that maximum
                    num_maxs=len(index_max_sum[0]) #Get how many slices exist with that maximum - sometimes more than 1
                    
                    slicenum=np.asarray(slicenum) #Convert list with slices to array
                    print("Slices with the most nodule pixels: {}".format(slicenum[-num_maxs:]))
                    print("Sum of nodule pixels in those slices is/are {}".format(max_sum))
                    print("\n")
                                        
                    #Save segmentation slice with nodule that corresponds to the same slice as original_CT slice
                    #We check and compare only with CT slices with nodules that were given
                    for num_slice in slicenum: #Loop over slices with max pixels - not just in slicenum[-num_maxs] since some images are outside of that range

                        for index,CTfile in enumerate(original_CTs): #Loop over original CT slices that have nodules
                            slice_number=CTfile.split('.')[4] #Get slice number from original CT file name
                            
                            #Get Slice number from dicom header instead of image name
                            get_slice=dicom.dcmread(input_path+'/'+CTfile) #Read file

                            if int(get_slice.InstanceNumber)==int(slice_number):#Just confirmation - should always be true

                                if num_slice==SEG_image_3D.shape[0]-int(slice_number):#Since slices here are in reverse order
                                                            
                                        SEG_image=SEG_image_3D[num_slice,:,:].copy() #Get a copy of the SEG slice with a nodule in the corresponding original CT slice

                                        thresh = cv2.threshold(SEG_image,0.5,255,cv2.THRESH_BINARY)[1] #Since a binary image (0s and 1s) set a threshold in the middle
                                        # cv2.imwrite(output_path+'/'+str(file[:-4])+'_SEG_no_box'+'.png',thresh) #Nodules without bounding box around them - values only 0 or 255
                                        
                                        SEG_image_color=cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB) #Convert back to RGB - values again only 0 or 255

                                        #Find contours of objects in that image
                                        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
                                        contours = contours[0] if len(contours) == 2 else contours[1]
                                        
                                        for cntr in contours: #Loop over contours and plot boxes in the colored image
                                            x,y,w,h = cv2.boundingRect(cntr)
                                            cv2.rectangle(SEG_image_color, (x, y), (x+w, y+h), (0, 0, 255), 1) 
                                            #Plot image below and not here since we may have many nodules and many slices that we want to plot
                                            
                                        #Save segmentation mask slice
                                        # #For colored-like version
                                        # plt.ioff() #Do not open figure window
                                        # plt.figure()
                                        # plt.imshow(SEG_image) 
                                        # plt.title("Slice {}".format(num_slice))
                                        
                                        #For segmentation masks activate imwrite below
                                        # if num_slice in slicenum[-num_maxs:]: #Loop over slices with maximum number of nodule pixels only
                                        #     cv2.imwrite(output_path+'/'+str(file[:-4])+'_slice_'+str(num_slice)+'_max'+'.png',SEG_image_color)
                             
                                        #     # #For a colored-like version - Binary image with box
                                        #     # plt.savefig(output_path+'/'+str(file[:-4])+'_'+'slice_'+str(num_slice)+'_max_colored'+'.png',dpi=dpi) #Colored nodule
                                          
                                        # else: #Also save the slices not with a maximum number of pixels. Some of them may be needed
                                        #     cv2.imwrite(output_path+'/'+str(file[:-4])+'_slice_'+str(num_slice)+'_not_max'+'.png',SEG_image_color)
                                            
                                        #     # #For a colored-like version
                                        #     # plt.savefig(output_path+'/'+str(file[:-4])+'_slice_'+str(num_slice)+'not_max_colored'+'.png',dpi=dpi)
                                                                                    
                                        # plt.close()
                                        
                                        annotation_CT=dicom.dcmread(input_path+'/'+annotated_CT_files[index]) #Read annotated_CT file
                                        annotation_CT_image=annotation_CT.pixel_array #Get its image
                                        CT_image=get_slice.pixel_array #Get image of original CT
                                        nodule=np.where(annotation_CT_image!=CT_image) #Find different pixels - nodule locations
                                        
                                        if np.sum(SEG_image_color[nodule])!=0 and len(contours)<2: #If nodule locations of annotated_CT exist in SEG file
                                        #and excluding SEG files if they have 2 or more nodules
                                        #Save SEG masks, original CT slices and annotated slices in corresponding order

                                            SEG_masks.append(file)
                                            original_CTs_final.append(CTfile)
                                            annotated_CTs_final.append(annotated_CT_files[index])                        
                                                            
                    if file not in SEG_masks: #Add file in errors if not added to the list above
                        SEG_masks_errors.append(file)      
                         
    if len(annotated_CT_files)-len(annotated_CTs_final)==1 and len(SEG_masks_errors)==1: #If only one file in errors then it should exist correspondance - add it to lists
         SEG_masks.append(SEG_masks_errors[0])
         original_CTs_final.append(list(set(original_CTs) - set(original_CTs_final))[0])
         annotated_CTs_final.append(list(set(annotated_CT_files) - set(annotated_CTs_final))[0]) 
                  
    if annotated_CTs_final==[] or original_CTs_final==[]:
        annotated_CTs_final=annotated_CT_files
        original_CTs_final=original_CTs
    
    end=time.time()
    print("Total time to find correspondance between SEG files to annotated and original CT slices (step3) was {} secs".format(end-start))
    print("\n")  
   
    if SEG_masks!=[]:
        if len(np.unique(annotated_CT_files))>len(np.unique(annotated_CTs_final)):
            print('REMARK: SEG Files Ignored since they are less or equal to annotated_CT files - {} vs {}'.format(len(np.unique(SEG_masks)),len(np.unique(annotated_CT_files))))
            sys.stdout.close()
            sys.stdout = sys.__stdout__ 
            return [],original_CTs,annotated_CT_files,[]
        else:
            sys.stdout.close()
            sys.stdout = sys.__stdout__ 
            return SEG_masks,original_CTs_final,annotated_CTs_final, SEG_masks_errors  
        
    else: #In case no SEG files exist
        sys.stdout.close()
        sys.stdout = sys.__stdout__ 
        return [],original_CTs,annotated_CT_files,[]
    
                                                

#1st function - extract CT, AI files, number of nodules, size of AI, CT and SEG files, and SEG errors
CTfiles, size_SEG, errors_SEG, AI_num_nods, size_CTs,size_AI, AI_slice_names =zip(*Parallel(n_jobs=-1)(delayed(CT_files_extract)(path,outputs[index],dpi=200) for index,path in enumerate(inputs)))

#2nd function - correspondance between annotated and original scans
original_CTs, annotated_CT_files, empty_CT_files, possible_nodules_excluded=zip(*Parallel(n_jobs=-1)(delayed(annotated_CTs_to_normal_scan)(path,CTfiles[index],outputs[index]) for index,path in enumerate(inputs)))

#3rd function - correspondance between SEG files, annotated_CT images, and original scan slices
SEG_masks,original_CTs_final,annotated_CTs_final,SEG_masks_errors=zip(*Parallel(n_jobs=-1)(delayed(mask_seg_slices)(original_CTs[index],path,outputs[index],annotated_CT_files[index],dpi=200) for index,path in enumerate(inputs)))


#Create dataframe with all participants and nodules
#Initialize columns to be filled below
column_names=['participant_id','AI_nod1','AI_nod2','AI_nod3','AI_nod4','AI_nod5','AI_nod6','AI_nod7','AI_nod8',
              'AI_nod9','AI_nod10','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
              '0-100tp','100-300tp','300+ tp','0-100fp','100-300fp','300+ fp','0-100fn','100-300fn','300+ fn']

df_all=pd.DataFrame(columns=column_names) #Initialize it


#Output File - Final computations
patient_names=[] #List to be filled with the patient IDs
all_volumes=[] #List to be filled with final list of volumes of nodules (TP and FP)
FP_num=[] #List to be filled with the number of FP findings for each patient

AI_pats={} #AI slices with nodules
AI_pats_vols={} #Volumes of AI slices with nodules
AI_pats_slices={} #All AI slices - original names
RedCap_pats={} #Slices with RedCap annotations
RedCap_pats_vols={} #Volumes of RedCap annotated slices
RedCap_ids={} #IDs just to be used for excel export at the end


for index_init,path in enumerate(inputs): #Loop over each patient  

    error_flag=0 #For cases with any errors that should be checked manually
    error_flag_images=0 #For cases with errors only in output images - only images should be checked manually
    distance_error=0 #Flag for cases with nearby TP nodules in which ID are confused.

    sys.stdout = open(outputs[index_init]+'output.txt', 'w') #Save output here
    
    patient_names.append(path.split('/')[-1]) #Add patient ID to this list
    
    #Load ground truth nodules in a dataframe and get an array of integers with the slices of the ground truth nodules
    try: #To ensure that manual annotations from REDCap exist
        ground_truth_nodules=pd.read_csv(ground_truth_path+path.split('/')[-1]+'.csv') #Read CSV file with REDCap annotations
        
        slice_cols=[col for col in ground_truth_nodules.columns if 'slice' in col] #Get list of column names with nodules
        slice_vals=ground_truth_nodules[slice_cols].values #Get list of slices with nodules
        slice_vals=slice_vals[~np.isnan(slice_vals)] #Exclude NaN values
        slice_vals=slice_vals.astype(int) #Convert them to integers
        print("Slices with nodules from REDCap are {}".format(slice_vals))
        RedCap_pats[path.split('/')[-1]]=slice_vals
        
        #Check first and last 10 slices of patients to see if GT slices with nodules exist there
        #If so, our algorithm may fail - In 137966 nodule in first 10 slices. The algorithm worked fine in that
        for j in slice_vals:
            if j<10 or j>size_CTs[index_init]-11: #between 20-30 slices exist
                print("In patient {} first 10 or last 10 slices contain nodule - Slice {}".format(path.split('/')[-1],j))
        
        ids=[col for col in ground_truth_nodules.columns if 'nodule_id' in col] #Get list of column names with nodule IDs
        nodule_ids=ground_truth_nodules[ids].values #Get list of nodule IDs
        nodule_ids=nodule_ids[~np.isnan(nodule_ids)] #Exclude NaN values
        nodule_ids=nodule_ids.astype(int) #Convert them to integers
        print("Nodule IDs are: {}".format(nodule_ids))
        RedCap_ids[path.split('/')[-1]]=nodule_ids
        
        vol_ids=[col for col in ground_truth_nodules.columns if 'volume_solid' in col] #Get list of column names with volume of solid nodules
        volume_ids=ground_truth_nodules[vol_ids].values #Get values of volumes of solid nodules
        temp_solid=volume_ids
        
        missing=np.where(np.isnan(volume_ids)) #Find where we don't have a value for volumes of solid nodules
        vol_ids_sub=[col for col in ground_truth_nodules.columns if 'volume_subsolid' in col] #Get list of column names with volume of subsolid nodules
        volume_ids_sub=ground_truth_nodules[vol_ids_sub].values #Get values of volumes of subsolid nodules
       
        #Since they are a list of list and we just want the inner list for volumes
        volume_ids=volume_ids[0]
        volume_ids_sub=volume_ids_sub[0]
        
        #If the solid component is <30mm3 we also keep the subsolid one. If it's >30mm3 we ignore subsolid, if exists. 
        #If we split based on groups (30-100 and/or 100-300mm3) this might result in having a few nodules belong to wrong volume subgroups.
        for ind, vol in enumerate(volume_ids):
            if vol<30 or np.isnan(vol)==True: #If volume of solid component <30mm3 or non-existent
                try:
                    if volume_ids_sub[ind]>=30 and np.isnan(vol)==False: #If solid component exists (is smaller than 30mm3) and subsolid >30mm3
                        volume_ids[ind]=vol+volume_ids_sub[ind]
                        print('Combined solid and subsolid components since volume of solid <30mm3')
                    elif volume_ids_sub[ind]>=30 and np.isnan(vol)==True: #If solid component doesn't exist and subsolid >30mm3
                        volume_ids[ind]=volume_ids_sub[ind]
                        print("Kept only subsolid component since there was no solid component")
                except:
                    pass
            
            if vol>=30: #If solid component >30mm3
                try: #To avoid error in 130781 without any subsolid components
                    if volume_ids_sub[ind]>0 and temp_solid[ind]>0: #Just to print information of subsolid component as well
                        print("Nodule with ID {} and volume {} has 2 components but only solid considered".format(nodule_ids[ind],temp_solid[ind]))
                        print('Volume of subsolid component is {}'.format(volume_ids_sub[ind]))                 
                except:
                    pass
        
        
        print("Volumes of the above nodules are: {}".format(volume_ids))
        print("\n")
        RedCap_pats_vols[path.split('/')[-1]]=volume_ids #Add volumes to the respective dictionary

    except: #If manual annotations file not found
        print("ERROR: No manual Annotations file from REDCap available")
        slice_vals=[]
        continue

    #Extract information from AI file
    try:
        df=pd.read_excel(AI_path,index_col=0) #Read Dataframe with AI nodules, using first column as indices
            
        for file in os.listdir(path): #Loop over all AI files and get information from first slice (0)
        
            #Last condition added to ensure that we get AI slice and not 'Results', as happened in 105179
            if len(file.split('.')[3])>1 and int(file.split('.')[4])==0 and int(file.split('.')[3])>=2000:
                dcm_file=dicom.dcmread(path+'/'+file)
                slice_location=float(dcm_file.SliceLocation)
                spacing=float(dcm_file.SpacingBetweenSlices)
                slice_number=int(dcm_file.InstanceNumber)
                assert slice_number==0
                break
            
        total_slices=size_AI[index_init] #total number of AI slices
        nod_locations=df.loc[int(path.split('/')[-1])][10:-1].values[df.loc[int(path.split('/')[-1])][10:-1].values!='-']
        #nod locations in the order of L01, L02 etc of AI detections           
        
        actual_slice=((-1/spacing)*nod_locations)+(slice_location/spacing)+total_slices #slice_number
        actual_slice=actual_slice.astype(float).round() #rounding
        
        AI_pats[path.split('/')[-1]]=actual_slice #Add slice number to AI_pats

        #Volumes of AI detections are extracted from a df created in aorta_calcium_lungnodules.ipynb
        volumes_AI=df.loc[int(path.split('/')[-1])][:10].values[df.loc[int(path.split('/')[-1])][:10].values!='-']
        AI_pats_vols[path.split('/')[-1]]=volumes_AI #Add volumes in AI_pats_vols
        print("Slices with nodules from AI are {}".format(AI_pats[path.split('/')[-1]]))
        print("Their volumes are: {}".format(volumes_AI))

    except: #If any error print it
        print(traceback.format_exc())
        pass

                    
    AI_pats_slices[path.split('/')[-1]]=AI_slice_names[index_init] #Add to AI_pats_slices all AI slice names
    print('\n')
    
    #Ensure that we have the same number of AI and original CT slices
    try:
        assert size_CTs[index_init]==len(AI_slice_names[index_init])==size_AI[index_init]
        print("Size of AI output scan is the same as size of original scan and equals to {}".format(size_AI[index_init]))
    except:
        print("ERROR! Num of AI slices is {} and num of original CT slices is {}".format(len(AI_slice_names[index_init]),size_CTs[index_init]))
        error_flag=1
        
    #Initialize empty lists to be filled below
    tp_final=[]
    fp_final=[]
    fn_final=[]
    tp_AI_final=[]
    
    vols_tp=[]
    vols_fp=[]
    vols_fn=[]
    vols_tp_AI=[]
    
    ids_tp=[] #IDs of GT
    ids_fp=[] #IDs of AI
    ids_fn=[] #IDs of GT
    ids_tp_AI=[] #IDs of AI

    same_slice_remove=[] #Add slices added incorrectly as TP from looping in nearby slices
    consecutive_slices=[] #Add slices here to avoid confusing nearby slices in which one might be TP and the other FN
    avoid_FP=[] #Avoid adding wrong slices to FP
    avoid_FP_volumes=[] #Add volumes of the above wrong slices
    error_FP=[] #For error in which FPs considered as TPs (check below)
    
    orig_check_sl=[ctname.split('.')[4] for ctname in original_CTs[index_init]] #Lists of strings containing possible slices with nodules
    
    print("Possible candidates containing nodules: {}".format(orig_check_sl)) #Some of them different from RedCap slices with +- a few
    print('\n')
    
    #It helps to have sorted list. We will avoid errors by checking in order. It changes eg. slice 40 to 040 and adds it in the beginning of list
    try:
        only_nums=[ctname.split('.')[4] for ctname in original_CTs[index_init]] #Get number of slice from full name
        sort_ind=[j for i in sorted(only_nums,key=int) for j in range(len(only_nums)) if only_nums[j]==i] #Get indices of sorted slices

        #Use the above indices and replace order of scan slices and annotations
        repl_original_CTs=[original_CTs[index_init][ind] for ind in sort_ind]
        repl_annotated_CT_files=[annotated_CT_files[index_init][ind] for ind in sort_ind]

        #Convert to lists, add them to the original lists (in the right index) and then convert to tuples to be used below in the loops
        or_list=list(original_CTs)         
        an_list=list(annotated_CT_files)
        or_list[index_init]=list(repl_original_CTs)
        an_list[index_init]=list(repl_annotated_CT_files)
        original_CTs=tuple(or_list)
        annotated_CT_files=tuple(an_list)
        
        print('Same but sorted',[ctname.split('.')[4] for ctname in original_CTs[index_init]])

    except: #If above gives errors, then lists are empty - change them to empty
        
        or_list=list(original_CTs)
        an_list=list(annotated_CT_files)
        or_list[index_init]=[]
        an_list[index_init]=[]
        original_CTs=tuple(or_list)
        annotated_CT_files=tuple(an_list)

    
    #Sort both GT and AI slices with nodules for consistency
    try:
        slice_vals,volume_ids,nodule_ids=zip(*sorted(zip(slice_vals,volume_ids,nodule_ids)))
        print("GT slices sorted ",slice_vals)
    except:
        pass
        
    try:
        AI_nodule_ids=[x+1 for x in range(len(AI_pats[path.split('/')[-1]]))] #Get AI nodule IDs
        AI_pats[path.split('/')[-1]],volumes_AI,AI_nodule_ids=zip(*sorted(zip(AI_pats[path.split('/')[-1]],volumes_AI,AI_nodule_ids)))
        print("AI slices sorted",AI_pats[path.split('/')[-1]])
    except:
        pass


    #To deal with error in eg. 748658 - confuses ID of nearby TP nodules - Now an error will be raised to check those cases manually
    distance_all=1000 #Set an initial distance between slices to a big number
    for ai_check in AI_pats[path.split('/')[-1]]: #Loop over AI slices
        for gt_check in slice_vals: #Loop over GT slices
            if np.abs(ai_check-gt_check)<=distance_all: #If the AI slice and the GT slice are close to each other (less than the defined distance)
                if np.abs(ai_check-gt_check)==distance_all: #If we have two times the same distance (eg. slices 373 and 379 from 376) then we might have error since we don't know which is the correct slice for that distance
                    error_flag=1 #Set error flag when there are two same AI or GT slices
                    distance_error=1 #Also set the error distance since even if error_flag set to 1 we will still have '!!!' in the created excel file
                else:
                    pass
                distance_all=np.abs(ai_check-gt_check) #Set the distance to the current one
        distance_all=1000 #When looping to next AI slice set it again to a big number to ensure that we also check the new AI slice with all the GT ones
      

    #Extraction of annotations (images) for radiologists to review
    
    #Initialize empty lists to be filled below with image data
    AI_images=[] #All AI images, TP and FP - Used just for counting
    AI_images_tp=[] #TP AI images - used for counting them
    AI_images_fp=[] #FP AI images
    AI_images_avoid_FP=[] #Keep track of slices since we may have many matches for same GT slice due to looping in a lot of nearby slices.
    
    CT_scans=[] #All original scans images, TP and FN
    CT_scans_tp=[] #TP nodule on radiologists' annotations
    CT_scans_fn=[] #FN nodule on radiologists' annotations
    CT_scans_same_slice=[] #To address cases of nodules considered as TP while they are FN - Look below

    AI_images_fp_slice=[] #These will contain the final FP images
    CT_scans_fn_slices=[] #These will contain the final FN images
    AI_images_tp_slice=[] #These will contain the final TP images
    
    #Only used to count if we have the correct numbers of images compared to findings - Not optimized to also extract TP images for now
    AI_images_tp_box=[]
    AI_images_avoid_tp_box=[]
    
 
    for GT_num,GT_slice in enumerate(slice_vals): #Loop over GT slices
        
        found=0 #An index to check if found=1 (TP) or not=0 (FN)
        
        for indexct,ctname in enumerate(original_CTs[index_init]): #Loop over possible CT files having a nodule
                
            for slice_CT_range in range(GT_slice-5,GT_slice+6): #Loop over +-5 CT files that may contain nodules since we may not have the same slice as in GT/REDCap

                #For 136470 with FP not getting in the loop below - there are FP not taken into account - this is why condition below added
                #'slice_CT_range>=0 and slice_CT_range<=size_CTs[index_init]' ensure that we don't have errors when nodule is on the first or last 10 slices of scan
                if int(ctname.split('.')[4])==slice_CT_range and slice_CT_range>=0 and slice_CT_range<=size_CTs[index_init]:
                    
                    for AI_num, slice_AI in enumerate(AI_pats[path.split('/')[-1]]): #Loop over slices in which AI detected a nodule
                        for slice_AI_range in range(int(slice_AI)-15,int(slice_AI)+16): #Loop over +-15 slices (changed from +-5 due to failure in 591162) of the nodule detected ones since we may have nodules in GT nearby

                            if int(slice_AI_range)==int(ctname.split('.')[4]): #If the AI slice is the same as one in CT file names
                                manual_CT=dicom.dcmread(path+'/'+ctname) #Load original CT DICOM slice to check for nodules
                                image_manual_CT=manual_CT.pixel_array #Load CT

                                manual_annotation=dicom.dcmread(path+'/'+annotated_CT_files[index_init][indexct]) #Load corresponding annotated slice
                                image_manual_annotation=manual_annotation.pixel_array #Load annotated CT

                                #Find locations of different pixels between original slice and annotations
                                differences_CT_annot=np.where(image_manual_CT!=image_manual_annotation) 

                                im_CT_annot=np.zeros((512,512)) #Initialize empty array with same dimensions as CT to be filled with only the annotated nodule
                               
                                im_CT_annot[differences_CT_annot]=image_manual_annotation[differences_CT_annot] #Keep only the region in which nodule exists in manual annotations
                
                                im_CT_annot[410:,468:]=0 #Set to zero bottom right region with 'F'
                                # cv2.imwrite(outputs[index_init]+'/'+ctname+'_manual_annotations.png',im_CT_annot) #Save annotated nodule only
            
                                im_CT_annot_thresh=cv2.threshold(im_CT_annot,1,255,cv2.THRESH_BINARY)[1] #Get thresholded version with 0 and 255 only        
                                
                                #####Extraction of annotations for radiologists to review   
                                image_manual_CT_new=cv2.normalize(image_manual_CT,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F) #Normalize image
                                new_type=image_manual_CT_new.astype(np.uint16) #Convert image to int16 to be used below - Below conversion to int16 may be omitted
                                all_new_type_fn=cv2.cvtColor((new_type).astype(np.uint16),cv2.COLOR_GRAY2RGB) #Convert grayscale image to colored
                                
                                contours=cv2.findContours(im_CT_annot_thresh.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Find contours
                                contours=contours[0] if len(contours)==2 else contours[1]
                                for cntr in contours: #Add rectangle around nodules
                                    x,y,w,h=cv2.boundingRect(cntr)
                                    cv2.rectangle(all_new_type_fn,(x,y),(x+w,y+h),(0,0,255),2)                               
                                #####Extraction of annotations for radiologists finishes here

                                                        
                                for AI_slice_num in AI_pats_slices[path.split('/')[-1]]: #Loop over all AI slices of that participant
                                    if int(slice_AI_range)==size_AI[index_init]-int(AI_slice_num.split('.')[4]): #If the AI slice is the same as one AI slice with nodule
                                                
                                        AI_dicom=dicom.dcmread(path+'/'+AI_slice_num) #Load AI DICOM slice
                                        image_AI=AI_dicom.pixel_array #Load AI CT slice
                                    
                                        #Resize AI image to (512,512) - same size as SEG and CT files below, convert to HSV and get mask for red and yellow
                                        AI_image=image_AI.copy() #As a good practice - to ensure that we don't change the original image
                                        AI_512=cv2.resize(AI_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC) #Resize to (512,512)
                                        AI_hsv=cv2.cvtColor(AI_512,cv2.COLOR_BGR2HSV) #Convert from BGR to HSV colorspace
                                        mask_im_red=cv2.bitwise_and(AI_hsv,AI_hsv, mask=cv2.inRange(AI_hsv, (100,0,0), (200, 255, 255))) #Red mask - lower and upper HSV values
                                        mask_im_red[0:50,:,:]=0 #Set top pixels that mention 'Not for clinical use' to zero - ignore them and keep only nodules

                                        #Until here mask_im_red is a color image with range of values from 0-255
                                        #Now we convert from BGR (emerged with bitwise operation) to grayscale to shape (512,512). This is used below to take a thresholded image
                                        mask_im_red_gray=cv2.cvtColor(mask_im_red,cv2.COLOR_BGR2GRAY) #Also changes range of values
                                
                                        #Get a smoothed (thresholded) image with only nodules
                                        #If 1 instead of 128 we have more pixels in the nodule contour and not well shaped
                                        mask_red_thresh = cv2.threshold(mask_im_red_gray,128,255,cv2.THRESH_BINARY)[1] 
                                        # cv2.imwrite(outputs[index_init]+'/'+ctname+'_AI_detections.png',mask_red_thresh) #Save AI nodules


                                        #####Extraction of annotations for radiologists to review
                                        image_manual_CT_new=cv2.normalize(image_manual_CT,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
                                        new_type=image_manual_CT_new.astype(np.uint16) #Convert image to int16 - Below conversion may be omitted
                                        all_new_type=cv2.cvtColor((new_type).astype(np.uint16),cv2.COLOR_GRAY2RGB) #Conversion to be easy to view

                                        contours=cv2.findContours(mask_red_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Find contours
                                        contours=contours[0] if len(contours)==2 else contours[1]
                                        for con_ind,cntr in enumerate(contours): #Add rectangles around nodules
                                            one_box=all_new_type.copy() #Get a copy of the image to add box around it without modifying the original one
                                            x,y,w,h=cv2.boundingRect(cntr)
                                            cv2.rectangle(one_box,(x,y),(x+w,y+h),(0,0,255),2)                                               
                                            
                                        image_manual_CT=np.zeros((512,512))
                                        #####Extraction of annotations for radiologists finishes here                                        
                               
                                        
                                        if np.where(mask_red_thresh!=0)[0].size==0: #If there are no AI detected nodules
                                            #For failure like in 369762, 985215, 184429, 335382 where TP while it should be FN
                                            #Works only when GT slice in orig_CT - It will fail in other cases
                                            if int(slice_CT_range)==int(GT_slice): #If the CT slice with possible nodules is the same as the GT nodule slice
                                                same_slice_remove.append(GT_slice) #This is a FN
                                                CT_scans_same_slice.append(all_new_type_fn) #Add to list of images considered as TP while they are FN


                                        if np.where(mask_red_thresh!=0)[0].size>0: #If there are AI detected nodules
                                            differences_AI_annot=np.where(im_CT_annot_thresh==mask_red_thresh) #Find where AI detected nodule overlaps with manual annotations
                                            #The above refers to the same nodule but with different contours from AI and manual annotations! Not to different nodule locations.
           
                                            im_overlap=np.zeros((512,512))  #Initialize empty array with same dimensions as CT to be filled with the overlap of AI and manually annotated nodule
                                            im_overlap[differences_AI_annot]=mask_red_thresh[differences_AI_annot] #Add overlap area of AI and manually detected nodule
                                            # cv2.imwrite(outputs[index_init]+'/'+ctname+str(slice_AI)+'_overlap.png',im_overlap)
                                            
                                            #For debugging 
                                            # print("GT_slice is {}".format(GT_slice))
                                            # print("slice_CT_range=ctname is {}".format(slice_CT_range))
                                            # print("slice_AI_range is {}".format(slice_AI_range))
                                            # print("AI_slice_num is {}".format(AI_slice_num))
                                            # print("slice_AI is {}".format(slice_AI))
                                                                                               
 
                                            if len(np.unique(im_overlap))>1: #If there is overlap between AI and manually annotated nodules
                                                    
                                                    #Since we may have many matches for same GT slice due to looping in a lot of nearby slices.
                                                    #Keep track of them and at the end keep only the one with the less distance from GT slice
                                                    avoid_FP.append(slice_AI) #Add to list to avoid adding a wrong FP - Failures in cases like 971099
                                                    avoid_FP_volumes.append(volumes_AI[AI_num]) #Keep track of the volume of these nodules - needed below
                                                    AI_images_avoid_FP.append(one_box) #Same for images
                                                    AI_images_avoid_tp_box.append([x,y,x+w,y+h]) #Add box location to list of TP to avoid
                                                    

                                                    #For failure in 673634 - AI detects two nodules on the same slice
                                                    #If we haven't added GT in TP, AI slice in TP_AI and GT in consecutive_slices (see below)
                                                    if GT_slice not in tp_final and slice_AI not in tp_AI_final and GT_slice not in consecutive_slices: 
                                                        
                                                        if slice_CT_range in slice_vals and GT_slice!=slice_CT_range: 
                                                            #If GT_slice is not the same as the CT_slice and CT_slice exists in GT don't do anything since that'a another nodule
                                                            #Get in here for 944714, 985215, 673634, 585377, 591162, 670208
                                                            pass

                                                        else: #Add GT to TP, AI_slice to TP_AI, same for their volumes and declare that nodule was found
                                                            
                                                            #Below compare AI volume with those of GT (that have the same slice) and keep the one that's it's closest to the AI. We assume that is the correct one
                                                            occurence2=np.where(np.array(slice_vals)==GT_slice) #Find where this slice occurs in GT slices
                                                            vol_AI_nod=volumes_AI[AI_num] #Get volume of AI nodules
                                                            big_dif=10000 #Initialize a value to a big number to keep track of difference between volumes
                                                            for i in occurence2[0]: #Loop over indices where slice can be found
                                                    
                                                                dif=np.abs(vol_AI_nod-volume_ids[i]) #Difference between volumes
                                                                if dif<=big_dif: #If difference smaller than the one defined                                                                   
                                                                    big_dif=dif #Set this difference as the new difference
                                                                    vol_gt_nod=volume_ids[i] #Get the volume of TP from REDCap
                                                                        
                                                            tp_final.append(GT_slice) #Add that slice to TP
                                                            vols_tp.append(vol_gt_nod) #Add volume to TP volumes
                                                            tp_AI_final.append(slice_AI) #Same for AI TP slice
                                                            vols_tp_AI.append(volumes_AI[AI_num]) #Same for AI TP volume

                                                            found=1 #Nodule found - not a FN    
                                                                                                                        
                                                            #Add corresponding images, slices and boxes to the respective lists
                                                            AI_images_tp.append(one_box) #TP image AI
                                                            CT_scans_tp.append(all_new_type) #TP image
                                                            AI_images_tp_slice.append(slice_AI) #AI TP slice
                                                            CT_scans.append(all_new_type) #List of all images
                                                            AI_images.append(one_box) #List of all AI images
                                                            AI_images_tp_box.append([x,y,x+w,y+h]) #AI TP box
                                                            

                                                    elif GT_slice not in tp_final and slice_AI in tp_AI_final and GT_slice not in consecutive_slices:
                                                        #If we haven't added GT in TP, in consecutive_slices (see below), but AI_slice was added to TP_AI
                                                        
                                                        if slice_CT_range in slice_vals and GT_slice!=slice_CT_range: #Failure in 673634
                                                            #If CT_slice in GT slices and GT_slice!=CT_slice don't do anything since it's another nodule
                                                            pass #Here for 944714,810826,985215,591162, 585377, 673634, 670208, 754238, 320656 - no effect so ok
                                                            
                                                        else: #Add GT to TP, AI_slice to TP_AI, same for their volumes, declare that nodule was found, and add to FP_errors (see below)
                                                                
                                                                tp_final.append(GT_slice) #Add TP slice to the TP list
                                                                
                                                                #Same process as above. Compare AI volume with those of REDCap for the same slice and keep the one closest to AI (assumed to be the correct one)
                                                                occurence2=np.where(np.array(slice_vals)==GT_slice) #Find where this slice occurs in GT slices
                                                                vol_AI_nod=volumes_AI[AI_num] #Get AI volumes
                                                                big_dif=10000 #initialize difference between volumes to a big value
                                                                for i in occurence2[0]: #Loop over indices where slice can be found
                                                                                                        
                                                                    dif=np.abs(vol_AI_nod-volume_ids[i]) #Get difference between GT and AI volume
                                                                    if dif<=big_dif: #If that difference smaller than the one set initially
                                                                        big_dif=dif #Replace the biggest difference with the current one
                                                                        vol_gt_nod=volume_ids[i] #Here we get the volume in REDCap closest to the one of AI
                                                                
                                                                vols_tp.append(vol_gt_nod) #Add REDCap volume found in the above loop to the TP volumes

                                                                #Below added for a second time but will be corrected below - need to have a correspondance/same number of slices as TP and that's why this is added
                                                                #Get here for 521779, 585377, 810826
                                                                tp_AI_final.append(slice_AI) #Add TP AI slice to list
                                                                vols_tp_AI.append(volumes_AI[AI_num]) #Same for TP AI volume

                                                                
                                                                #Here for 163557, 585377, 985215, 810826, 944714, 892519, 670208 but no errors created
                                                                error_FP.append(slice_AI) #Add AI slice to possible errors where FP considered as TP
                                                                found=1 #Nodule found - not a FN
                                                                
                                                                #Add images to the respective lists - Same as above
                                                                AI_images_tp.append(one_box) 
                                                                CT_scans_tp.append(all_new_type)     
                                                                AI_images_tp_slice.append(slice_AI)
                                                                CT_scans.append(all_new_type)
                                                                AI_images.append(one_box)
                                                                AI_images_tp_box.append([x,y,x+w,y+h])

                                                    #For failures in 670208 and 845594 with 2 same slices in GT
                                                    #If GT_slice in TP, GT_slice exists more than once in GT slices, and slice_AI not added in TP_AI
                                                    #Add them as above
                                                    elif GT_slice in tp_final and len(np.where(np.array(slice_vals)==GT_slice)[0])>1 and slice_AI not in tp_AI_final:
                                                        
                                                            tp_final.append(GT_slice) #Add GT slice to TP slices
                                                                            
                                                            #Same process as above
                                                            occurence2=np.where(np.array(slice_vals)==GT_slice) #Find where this slice occurs in GT slices
                                                            vol_AI_nod=volumes_AI[AI_num] #This is the AI volume
                                                            big_dif=10000
                                                            for i in occurence2[0]: #Loop over indices where slice can be found
                                                                        
                                                                dif=np.abs(vol_AI_nod-volume_ids[i])
                                                                if dif<=big_dif:
                                                                    big_dif=dif
                                                                    vol_gt_nod=volume_ids[i]

                                                            vols_tp.append(vol_gt_nod)
                                                            tp_AI_final.append(slice_AI)
                                                            vols_tp_AI.append(volumes_AI[AI_num])

                                                            found=1 #Nodule found - not a FN
                                                            
                                                            
                                                            #Add images to the respective lists
                                                            AI_images_tp.append(one_box)
                                                            CT_scans_tp.append(all_new_type)
                                                            AI_images_tp_slice.append(slice_AI)
                                                            CT_scans.append(all_new_type)
                                                            AI_images.append(one_box)
                                                            AI_images_tp_box.append([x,y,x+w,y+h])

                                            #Last condition len(np.where) to address failure in 429789 - Here when there is no overlap => no TP
                                            #Cases with TP and two same GT slices addressed exactly above
                                            elif GT_slice in tp_final and int(slice_CT_range)==int(GT_slice) and len(np.where(np.array(orig_check_sl)==str(GT_slice))[0])<2: 
                                            #If there is no overlap between AI and manual annotation (no TP) but GT_slice already in TP (added wrongly in another loop), CT_slice==GT_slice and we only have GT_slice once in possible CT slices with nodules
                                            #We get in here only for 136154 (two different annotations in two consecutive slices (one in AI detections (FP) and one in GT (FN))), 673634
                                            
                                                        ind_remove=np.where(np.array(tp_final)==GT_slice) #Find index of tp_final to remove
                                                        tp_final.remove(GT_slice) #Remove this slice from TP

                                                        del tp_AI_final[ind_remove[0][0]] #Remove it from TP_AI as well
                                                        del vols_tp[ind_remove[0][0]] #Remove the volume from TP volumes
                                                        del vols_tp_AI[ind_remove[0][0]] #Remove the volume from TP_AI volumes

                                                        fp_final.append(slice_AI) #Add AI_slice to FP
                                                        fn_final.append(GT_slice) #Add GT_slice to FN
                                                        vols_fp.append(volumes_AI[AI_num]) #Add volume AI to volumes with FPs
                                                        vols_fn.append(volume_ids[GT_num]) #Add volume of GT to volumes with FNs                                           
                                                        
                                                        #For failure in 673634 - two same slices in AI detections
                                                        consecutive_slices.append(GT_slice) #Add it in list since there is also a nearby slice in GT  
                                                        
                                                        #Remove image considered as TP and add the respective images as FP, FN
                                                        del AI_images_tp[ind_remove[0][0]]
                                                        del AI_images_tp_box[ind_remove[0][0]]
                                                        del CT_scans_tp[ind_remove[0][0]]
                                                        AI_images_fp.append(one_box)
                                                        CT_scans_fn.append(all_new_type_fn)
                                                        AI_images_fp_slice.append(slice_AI)
                                                        CT_scans_fn_slices.append(GT_slice) 
                                                        del AI_images_tp_slice[ind_remove[0][0]]

                                
                    #For failures in eg. 971099, 944714, 985215, 892519, 163557, 670208, 278319
                    #Since we may have many matches for same GT slice due to looping in a lot of nearby slices.
                    #Keep track of them all and at the end keep only the one with the less distance from GT slice
                    if len(avoid_FP)>1: #If we have more than one matches for the same slice
                        try: #Error since in 720754 no tp_AI_final[-1] exists yet - empty list - will be created when we loop over next slices
                            if [x for val_check in avoid_FP for x in range(int(val_check)-15,int(val_check)+15) if x==tp_AI_final[-1]]!=[]: 
                            #Above condition added since we may add slices to avoid_FP without actually adding a TP, meaning that we will replace wrong FP slice - example is the extra FP in 673634
        
                                # #For debugging
                                # print("GT_slice is",GT_slice)
                                # print('Possible slices with a TP_AI are {}'.format(avoid_FP)) 
                                # print('Their volumes are {}'.format(avoid_FP_volumes)) 
                   
                                distance=1000 #Initialize this value to be much bigger than the maximum difference of slices squared (here maximum is 15^2=225)
                                keep=-1 #initialize an index to -1
                                for ind_sel,select in enumerate(avoid_FP): #Loop over indices and values of wrong FP  
                                    
                                    #= set to correct error in 585377
                                    if np.abs(GT_slice-select)<=distance: #If the difference between our GT_slice and the possible wrong FP is less than the above distance
                                    
                                        #In case with 10 nodules (eg. 585377) we have GT in 131 and possible TP_AI in 129,133. == below added to find the correct one (133)                                                                        
                                         if np.abs(GT_slice-select)==distance: #In case that we have two AI slices with the same distance from GT slice
                                             ind_vol_GT=np.where(np.array(slice_vals)==GT_slice) #Get index where GT slice occurs to find GT volume

                                             for real_slice in ind_vol_GT[0]: #Loop over the GT occurences
                                                 
                                                 vol_ind_sel_dif=np.abs(avoid_FP_volumes[ind_sel]-volume_ids[real_slice]) #Volume difference of current AI slice with GT volume
                                                 vol_keep=np.abs(avoid_FP_volumes[keep]-volume_ids[real_slice]) #Volume difference of previous AI slice of same distance with GT volume
                                                 
                                                 if vol_ind_sel_dif>vol_keep: #If the current AI slice is further away in terms of volume size, ignore it
                                                     pass
                                    
                                                 elif vol_ind_sel_dif<=vol_keep: #If the current AI slice is closer in terms of volume size with GT volume, keep this instead
                                                     keep=ind_sel #Keep the index in that case
                                               
                                         else: #If np.abs(GT_slice-select)<distance
                                                distance=np.abs(GT_slice-select) #Replace distance with the new one
                                                keep=ind_sel #Keep the index 

                                        #Get in here also for the following: 335382, 384136, 395464, 427498, 440453,
                                        #591162, 673634, 944714, 971099, 985215, but no issue created
                                      
                                #In 985215 we get the correct results since we don't get in the if statement below and not replacement takes place
                                #AI slice 167 (AI5 nodule) matched with 169 of GT (L20) and then, AI slice 168 (AI6 nodule) matched with slice 170 of GT (L19) - since no replacement
                                print("Kept slice which is closest to GT slice - here {} with volume {}".format(avoid_FP[keep],avoid_FP_volumes[keep]))


                                #Below to address failure in 985215 after fixing 971099 and to avoid errors with TP_AI nodules and their volumes/IDs
                                #If the slice with the minimum distance not already added to TP_AI (whereas the corresponding GT was added to TP), add it
                                #This might happen if eg. we were not in any of the if/elif statements above (where avoid_FP was defined)
                                if avoid_FP[keep] not in tp_AI_final and GT_slice in tp_final: 
                                #We get in here at least for: 944714 985215 971099 335382 427498 440453 585377 591162 892519 163557 670208
                                    tp_AI_final[-1]=avoid_FP[keep] #Replace last TP AI slice
                                    
                                    #####Extraction of annotations for radiologists to review
                                    AI_images_tp[-1]=AI_images_avoid_FP[keep] #Replace TP image
                                    AI_images_tp_box[-1]=AI_images_avoid_tp_box[keep] #Change TP box coordinates
                                    AI_images_tp_slice[-1]=avoid_FP[keep] #Replace last slice with the slice from 
                                    #####Extraction of annotations for radiologists finishes here

                                    for FP_ind,FP_slice_check in enumerate(AI_pats[path.split('/')[-1]]):
                                        if avoid_FP[keep]==FP_slice_check and volumes_AI[FP_ind]==avoid_FP_volumes[keep]:
                                            vols_tp_AI[-1]=volumes_AI[FP_ind] #Replace AI volume with the correct one

                        except:
                            pass

 
                    avoid_FP=[] #Set the possible FP to empty for next GT slice loop
                    avoid_FP_volumes=[] #Same for their volumes
                    print('\n')
                    AI_images_avoid_FP=[] #And for the TP AI image
                        
 
        if found==0: #If there is no correspondance for nodule in the GT_slice with an AI nodule then it's a FN
            
            fn_final.append(GT_slice) #Add that to list of FN slices
            
            occurence2=np.where(np.array(slice_vals)==GT_slice) #Find where this slice occurs in GT slices
            
            #This will work only when we have two times the same slice. If more than two then it will fail
            for i in occurence2[0]: #Loop over indices where slice can be found
                if len(occurence2[0])==2 and GT_slice in tp_final: #If there are two same slices and one already added in TPs
                    if volume_ids[i] in vols_tp: #If the current volume added in TP volume then don't do anything
                        pass
                    else: #Otherwise added to FN
                        vols_fn.append(volume_ids[i])
                else: #If not, add the volume to FNs
                    vols_fn.append(volume_ids[GT_num])

            
            #####Extraction of annotations for radiologists to review
            flag=0 #Flag to check if we have added nodule to FN
            perfect_slice=0 #To denote that FN slice exists in CT slices with possible nodules
            
            #Sort to fix error in 998310 - better since it's more generalizable
            temp_original_CTs,temp_annot_CTs=original_CTs[index_init],annotated_CT_files[index_init] 
            #We can replace 'temp' with original ones - Already sorted above and they should work here - Does not make any difference if left untouched
    
            #Get index and slice number in which we have the FN in the above sorted list
            #This will not work if two FN in the same slice!
            if int(GT_slice) in np.array([int(ctname.split('.')[4]) for ctname in original_CTs[index_init]]):
                occurence=np.where(np.array([int(ctname.split('.')[4]) for ctname in temp_original_CTs])==int(GT_slice))[0][0] #Get indices to be accessed below
                perfect_slice=1 #indicate that slice with FN exists in CT slices with possible nodules
              
            for indexct,ctname in enumerate(temp_original_CTs): #Loop over possible CT files having a nodule
                    for slice_CT_range in range(GT_slice-5,GT_slice+6): #Loop over +-5 CT files that may contain nodules since we may not have the same slice as in GT/REDCap
                        if slice_CT_range>=0 and slice_CT_range<=size_CTs[index_init]: #To avoid errors if nodules in last/first 10 slices
                            
                            if perfect_slice==1: #If the slice with FN exists in CT slices with possible nodules, then check only this slice and not the rest
                                slice_CT_range=GT_slice #So that we won't loop in other slices
                                ctname=temp_original_CTs[occurence] #Change the name to be checked below to the current one
                                indexct=occurence #Same for the index
                            else:
                                pass   
                            

                            if flag==1: #If we have already added the FN in the list in a previous iteration break from the loop
                                break
                            
                            #For 136470 with FP not getting in the loop below - there are FP not taken into account - this is why condition below added
                            if int(ctname.split('.')[4])==slice_CT_range:

                                manual_CT=dicom.dcmread(path+'/'+ctname) #Load original CT DICOM slice to check for nodules
                                image_manual_CT=manual_CT.pixel_array #Load CT
                    
                                manual_annotation=dicom.dcmread(path+'/'+temp_annot_CTs[indexct]) #Load corresponding annotated slice
                                image_manual_annotation=manual_annotation.pixel_array #Load annotated CT
                    
                                #Find locations of different pixels between original slice and annotations
                                differences_CT_annot=np.where(image_manual_CT!=image_manual_annotation) 
                    
                                im_CT_annot=np.zeros((512,512)) #Initialize empty array with same dimensions as CT to be filled with only the annotated nodule
                               
                                im_CT_annot[differences_CT_annot]=image_manual_annotation[differences_CT_annot]
                                #Keep only the region in which nodule exists in manual annotations
                    
                                im_CT_annot[410:,468:]=0 #Set to zero bottom right region with 'F'
                                # cv2.imwrite(outputs[index_init]+'/'+ctname+'_manual_annotations.png',im_CT_annot) #Save annotated nodule only
                    
                                im_CT_annot_thresh=cv2.threshold(im_CT_annot,1,255,cv2.THRESH_BINARY)[1] #Get thresholded version with 0 and 255 only        
                                
                                image_manual_CT_new=cv2.normalize(image_manual_CT,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F) #Normalize image
                                new_type=image_manual_CT_new.astype(np.uint16) #Convert to int16 - Below not needed to convert to int16 again
                                all_new_type=cv2.cvtColor((new_type).astype(np.uint16),cv2.COLOR_GRAY2RGB) #Convert to color
                                
                                contours=cv2.findContours(im_CT_annot_thresh.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Find contours
                                contours=contours[0] if len(contours)==2 else contours[1]
                                for cntr in contours: #Add nodule contours to image
                                    x,y,w,h=cv2.boundingRect(cntr)
                                    cv2.rectangle(all_new_type,(x,y),(x+w,y+h),(0,0,255),2)                               
                                
                                if len(np.unique(new_type))>1:#Add images to corresponding lists              
                                    
                                    CT_scans_fn.append(all_new_type) #Original CT slice with nodule contour around the FN
                                    CT_scans.append(all_new_type) #Add image to the list of all images
                                    CT_scans_fn_slices.append(GT_slice) #Add slice to FN slice list

                                    flag=1 #Set flag to 1 to denote that nodule was added in FN findings

           
            #Set these to 0 again        
            perfect_slice=0
            flag=0
                    
            #####Extraction of annotations for radiologists finishes here
            
        print('\n')


    #Not all FP were taken into account above - We have mostly focussed until now to TP, FN
    #In cases like 136470 where FP not taken into account/added in the above lists we have the code below
    flag=0 #New flag here set to 0
    
    #Below to keep track of coordinates of FP nodule and slices
    blanks=[] #To keep track of temporary coordinates - modified below
    blanks_final=[] #Here the final coordinates
    blank_slices=[] #Here the final slices
    
    files_all=[] #Get a list of all the files in the folder of a particular participant
    for file in os.listdir(path):
        files_all.append(file)
    
    for AI_fp_inds,slice_with_FP in enumerate(AI_pats[path.split('/')[-1]]): #loop over all AI slices with nodules
        if slice_with_FP not in fp_final and slice_with_FP not in tp_AI_final: #If this slice not in FP and not in TP_AI
            fp_final.append(slice_with_FP) #Add it to FPs
            vols_fp.append(volumes_AI[AI_fp_inds]) #Add its volume to volumes of FPs

            #Failure in 892519 where the same two AI slices exist, one of them being TP and the other FP - Extracted images below will not be correct either
            #Similarly, in 673634 these two AI slices are both FP
            counting=0 #Initialize a counter to 0
            total_AI=len(np.where(np.array(AI_pats[path.split('/')[-1]])==slice_with_FP)[0]) #Find how many occurences of this slice
           
            #If this slice does not exist in TP_AI, it either exists in FP or not exists at all
            counting=counting+len(np.where(fp_final==slice_with_FP)[0]) #Similar as above but only with FP occurences now
            fp_position=[] #If we have the same FP slice more than once
            if counting<total_AI: #Get in here for 673634, 128443, 129311               
                #Similarly as above
                occurences=np.where(np.array(AI_pats[path.split('/')[-1]])==slice_with_FP)[0]
                for occur_ind,indexing in enumerate(occurences): #We may have the same FP slice more than once4
                    if volumes_AI[indexing] not in vols_fp: #Add volume and index
                        vols_fp.append(volumes_AI[indexing])
                        fp_position.append(occur_ind)
                             
                for i in range(total_AI-counting): #Add slice to FP list many times (as many as it occurs as FP)
                    fp_final.append(slice_with_FP)
                            
                                        
            #####Extraction of annotations for radiologists to review            
            for indexct_file,filename in enumerate(files_all): #Loop over list of all CT files
                if len(filename.split('.')[3])==1 and int(filename.split('.')[4])==int(slice_with_FP): #If we have an original CT slice and its slice number is in 'slice_with_FP'
                    
                    manual_CT=dicom.dcmread(path+'/'+filename) #Load original CT DICOM slice to check for nodules
                    image_manual_CT=manual_CT.pixel_array #Load CT                    
                    
                    for AI_slice_num in AI_pats_slices[path.split('/')[-1]]: #Loop over all AI slices of that participant
                        if int(slice_with_FP)==size_AI[index_init]-int(AI_slice_num.split('.')[4]) and flag==0: #If the AI slice is the same as one AI slice with nodule
                            # print("AI slice num {}".format(AI_slice_num))
                            # print("slice_with_FP {}".format(slice_with_FP))
                            # print("size_AI is {}".format(size_AI[index_init]))
                            AI_dicom=dicom.dcmread(path+'/'+AI_slice_num) #Load AI DICOM slice
                            image_AI=AI_dicom.pixel_array #Load AI CT slice
                        
                            #Resize AI image to (512,512) - same size as SEG and CT files below, convert to HSV and get mask for red and yellow
                            AI_image=image_AI.copy() #As a good practice - to ensure that we don't change the original image
                            AI_512=cv2.resize(AI_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC) #Resize to (512,512)
                            AI_hsv=cv2.cvtColor(AI_512,cv2.COLOR_BGR2HSV) #Convert from BGR to HSV colorspace
                            mask_im_red=cv2.bitwise_and(AI_hsv,AI_hsv, mask=cv2.inRange(AI_hsv, (100,0,0), (200, 255, 255))) #Red mask - lower and upper HSV values
                            mask_im_red[0:50,:,:]=0 #Set top pixels that mention 'Not for clinical use' to zero - ignore them and keep only nodules
        
                            #Until here mask_im_red is a color image with range of values from 0-255
                            #Now we convert from BGR (emerged with bitwise operation) to grayscale to shape (512,512)
                            #It is used below to take a thresholded image
                            mask_im_red_gray=cv2.cvtColor(mask_im_red,cv2.COLOR_BGR2GRAY) #Also changes range of values
                    
                            #Get a smoothed (thresholded) image with only nodules
                            #If 1 instead of 128 we have more pixels in the nodule contour and not well shaped
                            mask_red_thresh = cv2.threshold(mask_im_red_gray,128,255,cv2.THRESH_BINARY)[1] 
                            # cv2.imwrite(outputs[index_init]+'/'+ctname+'_AI_detections.png',mask_red_thresh) #Save AI nodules

                            #Normalize image and change its type to uint16
                            image_manual_CT_new=cv2.normalize(image_manual_CT,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)#,dtype=cv2.CV_32F)
                            new_type=image_manual_CT_new.astype(np.uint16)
                            all_new_type=cv2.cvtColor((new_type).astype(np.uint16),cv2.COLOR_GRAY2RGB)
                            
                            contours=cv2.findContours(mask_red_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                            contours=contours[0] if len(contours)==2 else contours[1]
                            
                            
                            #Below loop added to ensure that we catch all nodules and don't have missing slices
                            #This might happen for eg. 916856 where AI nodule 2 and 6 are on the same location (coordinates) but on different slices
                            for check_AI_slice in AI_pats[path.split('/')[-1]]: #Loop over AI slices
                                close_slice_found=0 #Initialize a flag to 0
                                check_AI_slice=int(check_AI_slice) #Get AI slice as integer
                                for check_AI_slice_range in range(check_AI_slice-10,check_AI_slice+10): #Loop over the previous and next 10 slices
                                    
                                    #If close_slice_found=0 then it's definitely a FP since there is no closeby GT slice there
                                    #If there is another AI slice within 10 slices from the one that is being checked or if the one being checked is in GT slices then we might have a TP as well 
                                    if (check_AI_slice_range in [int(x) for x in AI_pats[path.split('/')[-1]]] and check_AI_slice_range!=check_AI_slice) or check_AI_slice_range in [int(x) for x in slice_vals]:
                                        close_slice_found=1
                                    elif [int(x) for x in AI_pats[path.split('/')[-1]]].count(check_AI_slice)>1: #If this slice exists more than once in AI slices then we might also have a TP in them
                                        close_slice_found=1
                                        
                                            
                            for con_ind,cntr in enumerate(contours): #Plot nodule contours around it
                                one_box=all_new_type.copy()
                                x,y,w,h=cv2.boundingRect(cntr)
                                cv2.rectangle(one_box,(x,y),(x+w,y+h),(0,0,255),2)
                                # cv2.imwrite(outputs[index_init]+'/'+str(x)+','+str(y)+','+str(x+w)+','+str(y+h)+'_AI_detections.png',one_box)
                                
                                if close_slice_found==0: #it's definitely a FP
                                
                                    if len(np.unique(mask_red_thresh))>1 and flag==0: #If not already added, add it to FP slices          

                                        blanks.append([x,y,x+w,y+h]) #Append coordinates of FP nodule

                                        if len(blanks)==2: #If we have two nodules being FP 
                                        
                                            box_1=torch.tensor([blanks[-2]],dtype=torch.float) #Get box around first nodules
                                            box_2=torch.tensor([blanks[-1]],dtype=torch.float) #Same for second
                                            iou=float(bops.box_iou(box_1,box_2)) #calculate overlap between them
                                            
                                            if iou<0.05: #If there is almost no overlap then the new nodule is also FP
                                                AI_images.append(one_box) #Add image to list of images
                                                AI_images_fp.append(one_box) #Add image to list of FP images
                                                AI_images_fp_slice.append(slice_with_FP) #Add FP slice to list of FP
                                                
                                                blanks_final.append([x,y,x+w,y+h]) #Add also its coordinates to final ones
                                                blank_slices.append(slice_with_FP) #Add slice of it as well
                                                
                                                #If the slice being examined exists only once in AI slices and the next and the previous of it is not in the AI slice list then set flag to 1 to denote that nodule added in FP
                                                if len(np.where(np.array(AI_pats[path.split('/')[-1]])==slice_with_FP)[0])<=1 and (slice_with_FP+1 not in AI_pats[path.split('/')[-1]] and slice_with_FP-1 not in AI_pats[path.split('/')[-1]]): 
                                                    flag=1

                                                
                                        elif len(blanks)==1: #If this is the first box being examined add it to list of FPs
                                            AI_images.append(one_box)
                                            AI_images_fp.append(one_box)
                                            AI_images_fp_slice.append(slice_with_FP)
                                            blanks_final.append([x,y,x+w,y+h])
                                            blank_slices.append(slice_with_FP)
                                            
                                            #Same condition as above: If slice_with_FP+-1 not in AI slices with nodules (for error in 873698) - set flag to 1
                                            if len(np.where(np.array(AI_pats[path.split('/')[-1]])==slice_with_FP)[0])<=1 and (slice_with_FP+1 not in AI_pats[path.split('/')[-1]] and slice_with_FP-1 not in AI_pats[path.split('/')[-1]]): 
                                                flag=1
                                                
                                        else: #For 3 or more boxes:
                                            
                                            box_1=torch.tensor([blanks[-1]],dtype=torch.float) #Last nodule box
                                            
                                            flag_blanks_fin=0 #Set a flag to 0 (if remains 0 that would mean that this box/FP was not added yet)
                                            for fin_blank in blanks_final: #Loop over the boxes for which we are confident that they are FP
                                                box_check=torch.tensor([fin_blank],dtype=torch.float) #Get box of one of the confident FPs
                                                iou_check=float(bops.box_iou(box_1,box_check)) #Calculate overlap of it with the latest added box
                                                if iou_check>0.05: #here the opposite since if there is one match that means that it exists
                                                    flag_blanks_fin=1 #Set flag to 1 to denote that box (and so FP) already added in list                                                                                                         
                                            
                                            if flag_blanks_fin==0:#If this box not added yet then add it to FPs
                                                                                                        
                                                AI_images.append(one_box)
                                                AI_images_fp.append(one_box)
                                                AI_images_fp_slice.append(slice_with_FP)
                                                blanks_final.append([x,y,x+w,y+h])
                                                blank_slices.append(slice_with_FP)
                                                
                                                #Same condition as above to denote that box added
                                                if len(np.where(np.array(AI_pats[path.split('/')[-1]])==slice_with_FP)[0])<=1 and (slice_with_FP+1 not in AI_pats[path.split('/')[-1]] and slice_with_FP-1 not in AI_pats[path.split('/')[-1]]): 
                                                    flag=1
                                                    

                                else: #In case the slice might or might not be a FP

                                    if len(AI_images_tp_box)>0: #If we have already added some TP boxes of similar nodules in the corresponding variable
                                        
                                        #Everything below will only work for cases that we have added at least one TP image
                                        for tp_box in AI_images_tp_box: #loop over TP boxes
                                            box_tp=torch.tensor([tp_box],dtype=torch.float) #Get TP box
                                            box_new=torch.tensor([[x,y,x+w,y+h]],dtype=torch.float) #get current box
                                            iou_tp=float(bops.box_iou(box_tp,box_new)) #calculate overlap
                                            
                                            if iou_tp<0.01: #Means no overlap and so, two separate boxes - Now we are confident it's a FP
                                            
                                                #Now use same code as above where we were confident it was FP
                                                if len(np.unique(mask_red_thresh))>1 and flag==0: #If not already added, add it to FP slices          
                                                    
                                                    blanks.append([x,y,x+w,y+h]) #add box coordinates to list
 
                                                    if len(blanks)==2: # If two boxes added in list with boxes 
                                                    
                                                        box_1=torch.tensor([blanks[-2]],dtype=torch.float)
                                                        box_2=torch.tensor([blanks[-1]],dtype=torch.float)
                                                        iou=float(bops.box_iou(box_1,box_2))
                                                        
                                                        if iou<0.05:
                                                            AI_images.append(one_box)
                                                            AI_images_fp.append(one_box)
                                                            AI_images_fp_slice.append(slice_with_FP)
                                                            
                                                            blanks_final.append([x,y,x+w,y+h])
                                                            blank_slices.append(slice_with_FP)
                                                            
                                                            if len(np.where(np.array(AI_pats[path.split('/')[-1]])==slice_with_FP)[0])<=1 and (slice_with_FP+1 not in AI_pats[path.split('/')[-1]] and slice_with_FP-1 not in AI_pats[path.split('/')[-1]]): 
                                                                flag=1
                                  
                                                            
                                                    elif len(blanks)==1: #If only one box added in list with boxes
                                                        AI_images.append(one_box)
                                                        AI_images_fp.append(one_box)
                                                        AI_images_fp_slice.append(slice_with_FP)
                                                        blanks_final.append([x,y,x+w,y+h])
                                                        blank_slices.append(slice_with_FP)

                                                        #If slice_with_FP+-1 not in AI slices with nodules (for error in 873698) - set flag to 1
                                                        if len(np.where(np.array(AI_pats[path.split('/')[-1]])==slice_with_FP)[0])<=1 and (slice_with_FP+1 not in AI_pats[path.split('/')[-1]] and slice_with_FP-1 not in AI_pats[path.split('/')[-1]]): 
                                                            flag=1
                                                            
                                                    else: #If more than 2 boxes
                                                        box_1=torch.tensor([blanks[-1]],dtype=torch.float) #Get box 1
                                                        
                                                        flag_blanks_fin=0 #Set a flag to 0 (if remains 0 that would mean that this box/FP was not added yet)
                                                        for fin_blank in blanks_final:
                                                            box_check=torch.tensor([fin_blank],dtype=torch.float)
                                                            iou_check=float(bops.box_iou(box_1,box_check))
                                                            if iou_check>0.05: #here the opposite since if there is one match that means that it exists
                                                                flag_blanks_fin=1
                                                            else:
                                                                pass
                                                        
                                                        if flag_blanks_fin==0:
                                                            
                                                            AI_images.append(one_box)
                                                            AI_images_fp.append(one_box)
                                                            AI_images_fp_slice.append(slice_with_FP)
                                                            blanks_final.append([x,y,x+w,y+h])
                                                            blank_slices.append(slice_with_FP)
                                                            if len(np.where(np.array(AI_pats[path.split('/')[-1]])==slice_with_FP)[0])<=1 and (slice_with_FP+1 not in AI_pats[path.split('/')[-1]] and slice_with_FP-1 not in AI_pats[path.split('/')[-1]]): 
                                                                flag=1
                                     
                                    else: #Again we have exactly the same code as above for cases without TP boxes added
                                        if len(np.unique(mask_red_thresh))>1 and flag==0: #If not already added, add it to FP slices          
                                                        
                                            blanks.append([x,y,x+w,y+h])

                                            if len(blanks)==2:
                                            
                                                box_1=torch.tensor([blanks[-2]],dtype=torch.float)
                                                box_2=torch.tensor([blanks[-1]],dtype=torch.float)
                                                iou=float(bops.box_iou(box_1,box_2))
                                                
                                                if iou<0.05:
                                                    AI_images.append(one_box)
                                                    AI_images_fp.append(one_box)
                                                    AI_images_fp_slice.append(slice_with_FP)
                                                    
                                                    blanks_final.append([x,y,x+w,y+h])
                                                    blank_slices.append(slice_with_FP)
                                                    
                                                    if len(np.where(np.array(AI_pats[path.split('/')[-1]])==slice_with_FP)[0])<=1 and (slice_with_FP+1 not in AI_pats[path.split('/')[-1]] and slice_with_FP-1 not in AI_pats[path.split('/')[-1]]): 
                                                        flag=1
                                                    
                                            elif len(blanks)==1:
                                                AI_images.append(one_box)
                                                AI_images_fp.append(one_box)
                                                AI_images_fp_slice.append(slice_with_FP)
                                                blanks_final.append([x,y,x+w,y+h])
                                                blank_slices.append(slice_with_FP)
                                                #If slice_with_FP+-1 not in AI slices with nodules (for error in 873698) - set flag to 1
                                                if len(np.where(np.array(AI_pats[path.split('/')[-1]])==slice_with_FP)[0])<=1 and (slice_with_FP+1 not in AI_pats[path.split('/')[-1]] and slice_with_FP-1 not in AI_pats[path.split('/')[-1]]): 
                                                    flag=1
                                                    
                                            else:
                                         
                                                box_1=torch.tensor([blanks[-1]],dtype=torch.float)
                                                                              
                                                flag_blanks_fin=0
                                                for fin_blank in blanks_final:
                                                    box_check=torch.tensor([fin_blank],dtype=torch.float)
                                                    iou_check=float(bops.box_iou(box_1,box_check))
                                                    if iou_check>0.05: #here the opposite since if there is one match that means that it exists
                                                        flag_blanks_fin=1
                                                    else:
                                                        pass
                                                                                                
                                                if flag_blanks_fin==0:                                                    
                                                    
                                                    AI_images.append(one_box)
                                                    AI_images_fp.append(one_box)
                                                    AI_images_fp_slice.append(slice_with_FP)
                                                    blanks_final.append([x,y,x+w,y+h])
                                                    blank_slices.append(slice_with_FP)
                                                    if len(np.where(np.array(AI_pats[path.split('/')[-1]])==slice_with_FP)[0])<=1 and (slice_with_FP+1 not in AI_pats[path.split('/')[-1]] and slice_with_FP-1 not in AI_pats[path.split('/')[-1]]): 
                                                        flag=1
                                                            

                            #Set image and flag to zeros    
                            image_manual_CT=np.zeros((512,512))
                            flag=0



    #Avoid cases in which same slice added to FP images twice - We don't cover cases with three consecutive FP slices! 
    #Got in here for 136307,184429,335382, 873698
    
    these_keep=[] #list of indices to keep
    for ind,img in enumerate(AI_images_fp): #Loop over FP images
        num_dif=0 #Index to keep track of the same images contained in AI FP image list
        indices=[] #Keep track of indices of FP images
        for ind_last,img_last in enumerate(AI_images_fp): #Second loop on FP images
            if len(np.where(img!=img_last)[0])>5000: #For different FP images
                num_dif=num_dif+1 #Keep track of the number of different images in above list
                indices.append(ind_last) #and of their indices
                
        tot=len(AI_images_fp)-num_dif #Number of same images in the above list
        
        if tot==2: #If we have the same 2 FP images
            for i in range(len(AI_images_fp)):
                if i not in indices:
                    these_keep.append(i) #Indices to keep for that case - Get in here for 673634
                    
        elif tot==1: #+1 since we don't take into account ind_last=ind
            these_keep.append(ind)
    
    #Loop over the unique indices with FP, add FP to the corresponding list, and keep track of their slices
    AI_images_newfp=[]
    AI_images_newfp_slice=[]
    for kept in np.unique(these_keep):
        AI_images_newfp.append(AI_images_fp[kept])
        AI_images_newfp_slice.append(AI_images_fp_slice[kept])

    AI_images_fp=AI_images_newfp
    AI_images_fp_slice=AI_images_newfp_slice
    
    #####Extraction of annotations for radiologists finishes here            
   
    
    #For failure in 369762, 184429, 225969, 585377 where we get a TP while is should be an FN
    for ind_wrong,wrong_annot in enumerate(same_slice_remove): #Loop over slices that were accidentaly added in this list (slices added incorrectly as TP from looping in nearby slices)
    #We may have the same slice multiple times in same_slice_remove (and only once in TP)=> Not an issue since we will only get once inside the 'if' statement below
    #This list has slices with no overlap between GT and AI slices and for which slice_CT_range==GT_slice, meaning there are FN - Example 429703

        if wrong_annot in tp_final: #If this slice also in TP
            ind_remove=np.where(np.array(tp_final)==wrong_annot) #Find index of occurence
            tp_final.remove(wrong_annot) #Remove from TP

            #For failure in 810826 (defined the loop below for it), 985215, 369762
            #If got below it means that we had TP and TP_AI. The latter should also be removed and added to FP instead
            if tp_AI_final[ind_remove[0][0]] not in error_FP: #If the corresponding TP_AI slice not in error_FP
                fp_final.append(tp_AI_final[ind_remove[0][0]]) #Add it to FP
                fp_vol_check=np.where(np.array(AI_pats[path.split('/')[-1]])==tp_AI_final[ind_remove[0][0]])
                vols_fp.append(volumes_AI[fp_vol_check[0][0]])
                
                ##########Extraction of annotations for radiologists to review
                for indexct_file,filename in enumerate(files_all): #Loop over all files for that participant and get slice in which TP AI nodule can be found
                    if len(filename.split('.')[3])==1 and int(filename.split('.')[4])==int(tp_AI_final[ind_remove[0][0]]):
                        manual_CT=dicom.dcmread(path+'/'+filename) #Load original CT DICOM slice to check for nodules
                        image_manual_CT=manual_CT.pixel_array #Load CT

                for indexct_file,filename in enumerate(files_all):       
                    if int(size_AI[index_init]-tp_AI_final[ind_remove[0][0]])==int(filename.split('.')[4]) and len(filename.split('.')[3])!=1 and flag==0: #If the AI slice is the same as one AI slice with nodule

                            AI_dicom=dicom.dcmread(path+'/'+filename) #Load AI DICOM slice
                            image_AI=AI_dicom.pixel_array #Load AI CT slice
                        
                            #Resize AI image to (512,512) - same size as SEG and CT files below, convert to HSV and get mask for red and yellow
                            AI_image=image_AI.copy() #As a good practice - to ensure that we don't change the original image
                            AI_512=cv2.resize(AI_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC) #Resize to (512,512)
                            AI_hsv=cv2.cvtColor(AI_512,cv2.COLOR_BGR2HSV) #Convert from BGR to HSV colorspace
                            mask_im_red=cv2.bitwise_and(AI_hsv,AI_hsv, mask=cv2.inRange(AI_hsv, (100,0,0), (200, 255, 255))) #Red mask - lower and upper HSV values
                            mask_im_red[0:50,:,:]=0 #Set top pixels that mention 'Not for clinical use' to zero - ignore them and keep only nodules
        
                            #Until here mask_im_red is a color image with range of values from 0-255
                            #Now we convert from BGR (emerged with bitwise operation) to grayscale to shape (512,512)
                            #It is used below to take a thresholded image
                            mask_im_red_gray=cv2.cvtColor(mask_im_red,cv2.COLOR_BGR2GRAY) #Also changes range of values
                    
                            #Get a smoothed (thresholded) image with only nodules
                            #If 1 instead of 128 we have more pixels in the nodule contour and not well shaped
                            mask_red_thresh = cv2.threshold(mask_im_red_gray,128,255,cv2.THRESH_BINARY)[1] 
                            # cv2.imwrite(outputs[index_init]+'/'+ctname+'_AI_detections.png',mask_red_thresh) #Save AI nodules
                            
                            image_manual_CT_new=cv2.normalize(image_manual_CT,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)#,dtype=cv2.CV_32F)
                            new_type=image_manual_CT_new.astype(np.uint16)
                            
                            all_new_type=cv2.cvtColor((new_type).astype(np.uint16),cv2.COLOR_GRAY2RGB)
                            
                            contours=cv2.findContours(mask_red_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                            contours=contours[0] if len(contours)==2 else contours[1]
                            
                            for con_ind,cntr in enumerate(contours): #Plot contour around possible FP nodule
                                one_box=all_new_type.copy()
                                x,y,w,h=cv2.boundingRect(cntr)
                                cv2.rectangle(one_box,(x,y),(x+w,y+h),(0,0,255),2)
                                    
                                if len(np.unique(mask_red_thresh))>1 and flag==0: #Add image to FP images list        

                                    AI_images.append(one_box)
                                    AI_images_fp.append(one_box)
                                    AI_images_fp_slice.append(tp_AI_final[ind_remove[0][0]])
                                    
                                    #Set flag to 1 if this was the only nodule on that slice which was classified as TP whereas it was FP
                                    if len(np.where(np.array(AI_pats[path.split('/')[-1]])==tp_AI_final[ind_remove[0][0]])[0])<=1:
                                        flag=1

                                
                            image_manual_CT=np.zeros((512,512))
                            flag=0  
                            #####Extraction of annotations for radiologists finishes here
                
                
            del tp_AI_final[ind_remove[0][0]] #Then remove it from TP_AI
            del vols_tp_AI[ind_remove[0][0]] #Same for AI volume
            del AI_images[ind_remove[0][0]] #Delete a random of the same occuring images here just for same number below
            del AI_images_tp[ind_remove[0][0]] #Delete it from TP AI images
            del AI_images_tp_slice[ind_remove[0][0]] #Delete TP AI slice too
            del AI_images_tp_box[ind_remove[0][0]] #And TP AI box coordinates

            fn_final.append(wrong_annot) #Add it to FN
            vols_fn.append(vols_tp[ind_remove[0][0]]) #Add FN volume to list
            del vols_tp[ind_remove[0][0]] #Delete it from TP volumes
            
            CT_scans_fn.append(CT_scans_same_slice[ind_wrong]) #Add GT scan and slice to FN lists
            CT_scans_fn_slices.append(wrong_annot) #Add slice to FN slices

            del CT_scans_tp[ind_remove[0][0]] #Delete it from TPs

    
    try: #Confirm that we have the expected number of TPs, FPs and FNs
        assert len(tp_final)+len(fn_final)==len(slice_vals)
        assert len(tp_final)+len(fp_final)==len(AI_pats[path.split('/')[-1]])
    
    except: #If not print error - Should be checked by someone in that case
        print("ERROR!! File should be checked manually")
        print(traceback.format_exc())
        error_flag=1
        pass

    try: #Confirm that we have the expected number of TPs, FPs and FNs
        assert len(CT_scans_tp)+len(CT_scans_fn)==len(CT_scans)
        assert len(AI_images_tp)+len(AI_images_fp)==len(AI_images) #873698 gives error
    
    except: #If not print error - Should be checked by someone in that case
        print("ERROR in images!! File should be checked manually")
        print(traceback.format_exc())
        error_flag_images=1 
        pass
    
    try:
        assert len(CT_scans_tp)+len(CT_scans_fn)==len(tp_final)+len(fn_final)
        assert len(AI_images_tp)+len(AI_images_fp)==len(tp_AI_final)+len(fp_final)
    except:
        print("Should have error above if in here! File should be checked manually")
        error_flag_images=1 
        

    
    print('TP list is {}'.format(tp_final))
    print('FN list is {}'.format(fn_final))
    print('FP list is {}'.format(fp_final))
    print('TP_AI list is {}'.format(tp_AI_final))
    
    print('Volumes of TP are: {}'.format(vols_tp))
    print('Volumes of FN are: {}'.format(vols_fn))
    print('Volumes of FP are: {}'.format(vols_fp))
    print('Volumes of TP_AI are: {}'.format(vols_tp_AI))

    print('Num of CT_scans',len(CT_scans))    
    print('Num of CT_scans_tp',len(CT_scans_tp))
    print('Num of CT_scans_fn',len(CT_scans_fn))
    
    print('Num of AI_images',len(AI_images))
    print('Num of AI_images_tp',len(AI_images_tp))
    print('Num of AI_images_fp',len(AI_images_fp))
    
    #Initialize empty lists to keep track of indices of TP, FP, FN and TP_AI
    ids_tp_final=[]
    ids_fp_final=[]
    ids_fn_final=[]
    ids_tp_AI_final=[]


    #Vols and loops added for cases like 892519 with same AI slice two times - Should work for same GT slice as well  
    #Loop over all volumes of nodules and find their IDs    
    #Below flags are used since not possible to use 'break'    
    
    for tp_vol in vols_tp: #loop over TP volumes
        flag=0
        for tp in tp_final: #Loop over TP slices
            occurence=np.where(np.array(volume_ids)==tp_vol) #Find where this volume occurs in in volume_ids
            occurence2=np.where(np.array(slice_vals)==tp) #Find where this slice occurs in GT slices
            for i in occurence[0]: #Loop over indices where volume can be found
                if np.where(np.array(occurence2)==i)[1].size: #If the slice exists that contains that volume
                    for j in range(len(occurence2[0])): #Loop over occurences of that slice
                        if occurence2[0][j]==i:
                            if nodule_ids[occurence2[0][j]] not in ids_tp_final: #Find slice of the volume and confirm that is not in TP slices
                                if nodule_ids[occurence2[0][j]] not in ids_fn_final: #And not in FN slices
                                    if flag==0: #If not already added
                                        ids_tp_final.append(nodule_ids[occurence2[0][j]]) #Add it to TP slices
                                        flag=1 #Set flag to 1 to avoid adding it again
                                        
    for fp_vol in vols_fp: #Similar as above for FPs
        flag=0
        for fp in fp_final:
            occurence=np.where(np.array(volumes_AI)==fp_vol)
            occurence2=np.where(np.array(AI_pats[path.split('/')[-1]])==fp)
            for i in occurence[0]:
                if np.where(np.array(occurence2)==i)[0].size:
                    for j in range(len(occurence2[0])):
                        if occurence2[0][j]==i:
                            if AI_nodule_ids[occurence2[0][j]] not in ids_fp_final:
                                if AI_nodule_ids[occurence2[0][j]] not in ids_tp_AI_final:
                                    if flag==0:
                                        ids_fp_final.append(AI_nodule_ids[occurence2[0][j]])
                                        flag=1
                                       

    for fn_vol in vols_fn: #Similar as above for FNs
        flag=0
        for fn in fn_final:
            occurence=np.where(np.array(volume_ids)==fn_vol)
            occurence2=np.where(np.array(slice_vals)==fn)
            for i in occurence[0]:
                if np.where(np.array(occurence2)==i)[0].size:
                    for j in range(len(occurence2[0])):
                        if occurence2[0][j]==i:
                            if nodule_ids[occurence2[0][j]] not in ids_tp_final:
                                if nodule_ids[occurence2[0][j]] not in ids_fn_final:
                                    if flag==0:
                                        ids_fn_final.append(nodule_ids[occurence2[0][j]])
                                        flag=1

    for tp_vol_AI in vols_tp_AI: #Similar as above for TP_AI
        flag=0
        for tp_AI in tp_AI_final:
            occurence=np.where(np.array(volumes_AI)==tp_vol_AI)
            occurence2=np.where(np.array(AI_pats[path.split('/')[-1]])==tp_AI)
            for i in occurence[0]:
                if np.where(np.array(occurence2)==i)[0].size:
                    for j in range(len(occurence2[0])):
                        if occurence2[0][j]==i:
                            if AI_nodule_ids[occurence2[0][j]] not in ids_fp_final:
                                if AI_nodule_ids[occurence2[0][j]] not in ids_tp_AI_final:
                                    if flag==0:
                                        ids_tp_AI_final.append(AI_nodule_ids[occurence2[0][j]])
                                        flag=1 
                            

    print("IDs of TP are {}".format(ids_tp_final))
    print("IDs of FN are {}".format(ids_fn_final))
    print("IDs of FP are {}".format(ids_fp_final)) 
    print("IDs of TP_AI are {}".format(ids_tp_AI_final))

    try: #Confirm that we count all nodules
        assert len(ids_tp_final)==len(vols_tp)==len(tp_final)
        assert len(ids_fp_final)==len(vols_fp)==len(fp_final)
        assert len(ids_fn_final)==len(vols_fn)==len(fn_final)
        assert len(ids_tp_AI_final)==len(vols_tp_AI)==len(tp_AI_final)
    except:
        print("ERROR! Some nodule were missed or double counted! File should be checked manually")
        error_flag=1
    

    try:
        assert(len(tp_AI_final)+len(fp_final)==len(ids_tp_AI_final)+len(ids_fp_final)) 
    except:
        print("Error! Some errors in the above loop that tries to fix issue of same AI slice two times")


    print('\n')
   
    if len(tp_AI_final)!=len(tp_final):
        print("Error! List tp_AI_final does not contain only matching TP of radiologists or contains fewer than them! File should be checked manually")
        error_flag=1
    
    print('\n')

    print('-----------------------------------------------------------------------------------')
    print("\n")
                                                          
    #Plot FP and FN slices with nodule contour around them
    #Cases 440453 confuse FP and TP in consecutive slices and 278319 similarly 

    for ind_fn,one_box in enumerate(CT_scans_fn): #Loop over FN images
        plt.ioff()
        plt.figure()
        plt.imshow(one_box) 
        plt.title(str(int(CT_scans_fn_slices[ind_fn])))
        
        ind=0 #Here for same slices to avoid overlapping previously saved image
        while os.path.exists(outputs[index_init]+'/'+str(int(CT_scans_fn_slices[ind_fn]))+'fnlastfinal_gray'+str(ind)+'.png'):
            ind=ind+1 #If more than one FN in same slice then name them 1,2 etc.
        while os.path.exists(outputs[index_init]+'/0'+str(int(CT_scans_fn_slices[ind_fn]))+'fnlastfinal_gray'+str(ind)+'.png'):
            ind=ind+1 #If more than one FN in same slice then name them 1,2 etc.
        while os.path.exists(outputs[index_init]+'/00'+str(int(CT_scans_fn_slices[ind_fn]))+'fnlastfinal_gray'+str(ind)+'.png'):
            ind=ind+1 #If more than one FN in same slice then name them 1,2 etc.

        try:
            if vols_fn[ind_fn]>=30 and len(CT_scans_fn)==len(vols_fn): #If vol>30 and we don't have any image errors
                if int(CT_scans_fn_slices[ind_fn])<100: #To have them ordered to be reviewed faster
                    if int(CT_scans_fn_slices[ind_fn])<10:
                        plt.savefig(outputs[index_init]+'/00'+str(int(CT_scans_fn_slices[ind_fn]))+'fnlastfinal_gray'+str(ind)+'.png',dpi=200)
                    else:
                        plt.savefig(outputs[index_init]+'/0'+str(int(CT_scans_fn_slices[ind_fn]))+'fnlastfinal_gray'+str(ind)+'.png',dpi=200) 
                else:
                    plt.savefig(outputs[index_init]+'/'+str(int(CT_scans_fn_slices[ind_fn]))+'fnlastfinal_gray'+str(ind)+'.png',dpi=200) 
        except:
            if len(CT_scans_fn)!=len(vols_fn): #For error cases save all images since they will be checked manually
                if int(CT_scans_fn_slices[ind_fn])<100: #To have them ordered to be reviewed faster
                    if int(CT_scans_fn_slices[ind_fn])<10:
                        plt.savefig(outputs[index_init]+'/00'+str(int(CT_scans_fn_slices[ind_fn]))+'fnlastfinal_gray'+str(ind)+'.png',dpi=200)
                    else:
                        plt.savefig(outputs[index_init]+'/0'+str(int(CT_scans_fn_slices[ind_fn]))+'fnlastfinal_gray'+str(ind)+'.png',dpi=200) 
                else:
                    plt.savefig(outputs[index_init]+'/'+str(int(CT_scans_fn_slices[ind_fn]))+'fnlastfinal_gray'+str(ind)+'.png',dpi=200)
                
        plt.close()  
        
        
    for ind_fp,one_box in enumerate(AI_images_fp): #Same for FP images
        plt.ioff()
        plt.figure()
        plt.imshow(one_box) 
        plt.title(str(int(AI_images_fp_slice[ind_fp])))
        
        ind=0
        while os.path.exists(outputs[index_init]+'/'+str(int(AI_images_fp_slice[ind_fp]))+'fpAIlastfinal_gray'+str(ind)+'.png'):
            ind=ind+1
        while os.path.exists(outputs[index_init]+'/0'+str(int(AI_images_fp_slice[ind_fp]))+'fpAIlastfinal_gray'+str(ind)+'.png'):
            ind=ind+1
        while os.path.exists(outputs[index_init]+'/00'+str(int(AI_images_fp_slice[ind_fp]))+'fpAIlastfinal_gray'+str(ind)+'.png'):
            ind=ind+1


        try: #Since when we have errors we might have more FP images, and therefore we won't be able to access ind_fp in vols
            if vols_fp[ind_fp]>=30 and len(AI_images_fp)==len(vols_fp): #If vol>30 and we don't have any image errors
                if int(AI_images_fp_slice[ind_fp])<100: #To have them ordered to be reviewed faster
                    if int(AI_images_fp_slice[ind_fp])<10:
                        plt.savefig(outputs[index_init]+'/00'+str(int(AI_images_fp_slice[ind_fp]))+'fpAIlastfinal_gray'+str(ind)+'.png',dpi=200) 
                    else:
                        plt.savefig(outputs[index_init]+'/0'+str(int(AI_images_fp_slice[ind_fp]))+'fpAIlastfinal_gray'+str(ind)+'.png',dpi=200) 
                else:
                    plt.savefig(outputs[index_init]+'/'+str(int(AI_images_fp_slice[ind_fp]))+'fpAIlastfinal_gray'+str(ind)+'.png',dpi=200) 

        except:
            if len(AI_images_fp)!=len(vols_fp): #For error cases save all images since they will be checked manually
                if int(AI_images_fp_slice[ind_fp])<100: #To have them ordered to be reviewed faster
                    if int(AI_images_fp_slice[ind_fp])<10:
                        plt.savefig(outputs[index_init]+'/00'+str(int(AI_images_fp_slice[ind_fp]))+'fpAIlastfinal_gray'+str(ind)+'.png',dpi=200)
                    else:
                        plt.savefig(outputs[index_init]+'/0'+str(int(AI_images_fp_slice[ind_fp]))+'fpAIlastfinal_gray'+str(ind)+'.png',dpi=200) 
                else:
                    plt.savefig(outputs[index_init]+'/'+str(int(AI_images_fp_slice[ind_fp]))+'fpAIlastfinal_gray'+str(ind)+'.png',dpi=200) 
        
        
        plt.close()       
        

        
    #Add information about nodules (id and volume) to dataframe 
    
    dict_add=dict.fromkeys(column_names) #Get column names
    dict_add['participant_id']=path.split('/')[-1] #get participant_id
    
    small_nodule_flag=0 #To not take into account findings smaller than 30mm3 when calculating number of TP, FP, and FN
    
    if error_flag==0 or distance_error==1: #If no errors
    
        try: #Since we might have some errors if above process didn't work for that participant

            #Initialize total number of findings in each volume subgroup for each of TP, FP, and FN to 0
            tp_100=0
            tp_100_300=0
            tp_300plus=0
            fp_100=0
            fp_100_300=0
            fp_300plus=0
            fn_100=0
            fn_100_300=0
            fn_300plus=0
            
            for ai_ind,ai_id in enumerate(ids_tp_AI_final): #Loop over TP and fill corresponding fields
                dict_add['AI_nod'+str(ai_id)]=int(tp_AI_final[ai_ind]) #AI nodule ID
                dict_add['V'+str(ai_id)]=vols_tp_AI[ai_ind] #Volume of tha nodule
                
                #Those below could also added in the 'ids_tp_final' loop below
                if float(vols_tp[ai_ind])<=100 and float(vols_tp[ai_ind])>=30:
                    tp_100=tp_100+1
                elif float(vols_tp[ai_ind])>100 and float(vols_tp[ai_ind])<=300:
                    tp_100_300=tp_100_300+1
                elif float(vols_tp[ai_ind])>300:
                    tp_300plus=tp_300plus+1
                else:
                    print("Error! For TP, Volume in GT <30mm3 and equal to {}mm3. Might have subsolid component as well that wasn't considered".format(float(vols_tp[ai_ind])))
                    print('\n')
                
            
            for ai_ind_fp,ai_id_fp in enumerate(ids_fp_final): #Same as above for FPs
                dict_add['AI_nod'+str(ai_id_fp)]=int(fp_final[ai_ind_fp])
                dict_add['V'+str(ai_id_fp)]=vols_fp[ai_ind_fp]  
                
                if float(vols_fp[ai_ind_fp])<=100 and float(vols_fp[ai_ind_fp])>=30:
                    fp_100=fp_100+1
                elif float(vols_fp[ai_ind_fp])>100 and float(vols_fp[ai_ind_fp])<=300:
                    fp_100_300=fp_100_300+1
                elif float(vols_fp[ai_ind_fp])>300:
                    fp_300plus=fp_300plus+1
                else:
                    print("FP nodule with volume {}mm3 not taken into account".format(float(vols_fp[ai_ind_fp])))
                    print('\n')
                    small_nodule_flag=1
                
                
            for ai_ind_tp,ai_id_tp in enumerate(ids_tp_final): #Here only to get TP_AI ID and fill corresponding column
                ai_ind=ids_tp_AI_final[ai_ind_tp]
                dict_add['AI_nod'+str(ids_tp_AI_final[ai_ind_tp])]=str(dict_add['AI_nod'+str(ids_tp_AI_final[ai_ind_tp])])+' - L'+str(ai_id_tp)
            
            
            for ind_fn,id_fn in enumerate(ids_fn_final): #Same for FNs
                if float(vols_fn[ind_fn])<=100 and float(vols_fn[ind_fn])>=30:
                    fn_100=fn_100+1
                elif float(vols_fn[ind_fn])>100 and float(vols_fn[ind_fn])<=300:
                    fn_100_300=fn_100_300+1
                elif float(vols_fn[ind_fn])>300:
                    fn_300plus=fn_300plus+1
                else:
                    print("Error! For FN, Volume in GT <30mm3 and equal to {}mm3".format(float(vols_fn[ind_fn])))
                    print('\n')
        
            
            #Add all the above calculated values to the proper names in dictionary
            dict_add['0-100tp']=tp_100
            dict_add['100-300tp']=tp_100_300
            dict_add['300+ tp']=tp_300plus
            dict_add['0-100fp']=fp_100
            dict_add['100-300fp']=fp_100_300
            dict_add['300+ fp']=fp_300plus
            dict_add['0-100fn']=fn_100
            dict_add['100-300fn']=fn_100_300
            dict_add['300+ fn']=fn_300plus
            
            #If no TP, FP, and FNs then fill in position 'AI_nod1' that there were no nodules
            if tp_100==0 and tp_100_300==0 and tp_300plus==0 and fp_100==0 and fp_100_300==0 and fp_300plus==0 and small_nodule_flag==0:
                dict_add['AI_nod1']='nonods'
            
        except: #Don't fill anything if any errors for that participant
            pass
    
    #If error in images only (or error with two same AI or GT images) then add '!!!' - Might also confuse TP ids when distance error
    if (error_flag_images==1 and error_flag==0) or distance_error==1:
        dict_add['AI_nod1']='!!!'+str(dict_add['AI_nod1'])
        
    #if any error with TP, FP, or FN then add 'xxx'
    if error_flag==1 and error_flag_images==1:
        dict_add['AI_nod1']='xxx'+str(dict_add['AI_nod1'])

    
    df_all=df_all.append(dict_add,ignore_index=True) #Add the dictionary to dataframe


    print('-----------------------------------------------------------------------------------')
    print("\n")



    #All below just for print AI_slices, original_CT_slices, annotations and SEG files
    #Some prints were commented since they were incorrect, like 'Ground truth is'...

    slice_vals_found=[] #Empty lists to be filled with the slice values from REDCap that were detected by AI below
    slice_vals=np.array(slice_vals) #Since for case 757591 we get a tuple and so, an error below
        
    #Get which detections of AI correspond to which slices of the original CT scan
    #There are also FP detections from AI that are not taken into account here
    detectionsCT=[]
    detectionsAI=[]

    if list(AI_num_nods[index_init].keys())!=[]: #If there are slices with nodules in AI outputs
        for AI_file in AI_num_nods[index_init].keys(): #Loop over AI slices with nodules
            for orig_CT_slice in original_CTs[index_init]: #Loop over original CT slices with nodules
                if int(AI_file.split('.')[4])==size_CTs[index_init]-int(orig_CT_slice.split('.')[4]): #If there is correspondance between slices of original CT and AI output (reverse order) then we have a match
                    detectionsCT.append(orig_CT_slice) #Add original CT slice to list
                    detectionsAI.append(AI_file) #Add AI output slice to list
                    
    else: #If there are no AI detections
        if slice_vals.size!=0: #And if there are manual annotations
            print('IMPORTANT!: There no AI detections but there are manual annotations')
        else:
            print("IMPORTANT!: There are no AI detections and there are no manual annotations")
        print('\n')

            
    #If we have AI outputs, print the correspondance for all (SEG_files, AI outputs, Original CT slices and Annotated CT slices)
    if list(AI_pats[path.split('/')[-1]])!=[]: 

        for ind1,CT in enumerate(detectionsCT): #Loop over original CT slices for which we have AI detections
            for ind2, orig_CT in enumerate(original_CTs_final[index_init]): #Loop over a similar list of original slices for which we have AI detections but with a different order (index is used below so that there is correspondance)
                
                if CT==orig_CT: #If we have the same CT slice in both lists

                    if int(original_CTs_final[index_init][ind2].split('.')[4]) in slice_vals: #Check if this is an actual nodule
                            try:
                                for i in np.where(slice_vals==int(original_CTs_final[index_init][ind2].split('.')[4]))[0]: #Loop over indices of REDCap slices that exist in annotated CTs as well
                                    nod_ids=int(nodule_ids[i]) #Get IDs of these nodules from REDCap (L01 etc. in Syngo.via manual annotations)
                                    volumes=float(volume_ids[i]) #Get volumes of these nodules
                                    # print("Ground truth: The volume of this is {}, the ID in manual annotations is {}, and the slice is {}".format(volumes,nod_ids,np.unique(slice_vals[np.where(slice_vals==int(original_CTs_final[index_init][ind2].split('.')[4]))])))
                            except:
                                pass
                            
                            slice_vals_found.append(int(original_CTs_final[index_init][ind2].split('.')[4])) #True nodule found
    
                            try: #If SEG file exists print correspondance
                                # print(colored('Segmentation mask is: {}, slice {}', 'green').format(SEG_masks[index_init][ind2],detectionsAI[ind1].split('.')[4]))
                                print('Segmentation mask is: {}, slice {}'.format(SEG_masks[index_init][ind2],detectionsAI[ind1].split('.')[4]))

                            except IndexError:
                                print("No SEG mask available")
                                pass
                            # print(colored('Annotated CT is: {}', 'green').format(annotated_CTs_final[index_init][ind2]))
                            # print(colored('Original CT image is: {}', 'green').format(original_CTs_final[index_init][ind2]))
                            # print(colored('AI output image with nodules is: {}', 'green').format(detectionsAI[ind1]))
                            print('Annotated CT is: {}'.format(annotated_CTs_final[index_init][ind2]))
                            print('Original CT image is: {}'.format(original_CTs_final[index_init][ind2]))
                            print('AI output image with nodules is: {}'.format(detectionsAI[ind1]))
                            print("\n")
                            
                    else: #possible FP - maybe also be TP that extends from the slice that exists in slice_vals
                        for i in slice_vals: #Check slices close to slice_vals to see if we have annotations - maybe also be TP but for now added in FP
                            if int(original_CTs_final[index_init][ind2].split('.')[4]) in range(i-1,i-5,-1) or int(original_CTs_final[index_init][ind2].split('.')[4]) in range(i+1,i+6):
                                # print("High chances of having TP (even though not same slice as in REDCap or no annotation file available) for the following:")
                                try:
                                    nod_ids=int(nodule_ids[np.where(slice_vals==int(original_CTs_final[index_init][ind2].split('.')[4]))])
                                    volumes=float(volume_ids[np.where(slice_vals==int(original_CTs_final[index_init][ind2].split('.')[4]))])
                                    # print("If so, the volume of this is {} and the ID in manual annotations is {} and slice is {}".format(volumes,nod_ids,slice_vals[np.where(slice_vals==int(original_CTs_final[index_init][ind2].split('.')[4]))]))
                                except:
                                    pass
                    
                        try:
                            # print(colored('Segmentation mask is: {}, slice {}', 'green').format(SEG_masks[index_init][ind2],detectionsAI[ind1].split('.')[4]))
                            print('Segmentation mask is: {}, slice {}'.format(SEG_masks[index_init][ind2],detectionsAI[ind1].split('.')[4]))

                        except IndexError:
                            print("No SEG mask available")
                            pass
                        # print(colored('Annotated CT is: {}', 'green').format(annotated_CTs_final[index_init][ind2]))
                        # print(colored('Original CT image is: {}', 'green').format(original_CTs_final[index_init][ind2]))
                        # print(colored('AI output image with nodules is: {}', 'green').format(detectionsAI[ind1]))
                        print('Annotated CT is: {}'.format(annotated_CTs_final[index_init][ind2]))
                        print('Original CT image is: {}'.format(original_CTs_final[index_init][ind2]))
                        print('AI output image with nodules is: {}'.format(detectionsAI[ind1]))
                        print("\n")
         
        for ind3, CT_file in enumerate(original_CTs_final[index_init]): #Print files with nodules not found by the AI
            if CT_file not in detectionsCT: #Since it contains only slices that were also found by the AI
                if int(original_CTs_final[index_init][ind3].split('.')[4]) in slice_vals: #Check if this is an actual nodule
                    slice_vals_found.append(int(original_CTs_final[index_init][ind3].split('.')[4])) #To make sure that we have taken it into account
                    # print('True nodule not detected by AI:')
                    try:
                        # print(colored('Segmentation mask is: {}', 'green').format(SEG_masks[index_init][ind3]))
                        print('Segmentation mask is: {}'.format(SEG_masks[index_init][ind3]))

                    except IndexError:
                        pass

                    # print(colored('Annotated CT is: {}', 'green').format(annotated_CTs_final[index_init][ind3]))
                    # print(colored('Original CT image is: {}', 'green').format(original_CTs_final[index_init][ind3]))
                    print('Annotated CT is: {}'.format(annotated_CTs_final[index_init][ind3]))
                    print('Original CT image is: {}'.format(original_CTs_final[index_init][ind3]))
                    print("\n")

                else:
                    # print("No true nodule and not detected by AI or no annotation file available")
                    try:                 
                        # print(colored('Segmentation mask is: {}', 'green').format(SEG_masks[index_init][ind3]))
                        print('Segmentation mask is: {}'.format(SEG_masks[index_init][ind3]))

                    except:
                        pass
                    
                    # print(colored('Annotated CT is: {}', 'green').format(annotated_CTs_final[index_init][ind3]))
                    # print(colored('Original CT image is: {}', 'green').format(original_CTs_final[index_init][ind3]))
                    print('Annotated CT is: {}'.format(annotated_CTs_final[index_init][ind3]))
                    print('Original CT image is: {}'.format(original_CTs_final[index_init][ind3]))
                    print("\n")
           
        #We never get in the loop below - Confirmed
        for ind4, CT_file_rest in enumerate(original_CTs[index_init]): #Print files with nodules not found by the AI and not having SEG file
            if CT_file_rest not in detectionsCT and CT_file_rest in slice_vals: #CT_file_rest not in original_CTs_final[index_init] and 
                print("ERROR!! We didn't expect to be here!") #Confirmed
                # print(colored('Annotated CT is: {}', 'green').format(annotated_CT_files[index_init][ind4]))
                # print(colored('Original CT image is: {}', 'green').format(original_CTs[index_init][ind4]))
                print('Annotated CT is: {}'.format(annotated_CT_files[index_init][ind4]))
                print('Original CT image is: {}'.format(original_CTs[index_init][ind4]))
                print("\n")
    
    #We should only be in here when there are manual annotations but no AI detections (FN) 
    else: #otherwise print only SEG files, annotated CT images and original CT slices (if no AI outputs) - only for SEG files that exist

        for index in range(len(original_CTs_final[index_init])): 
            try:
                # print(colored('Segmentation mask is: {}', 'green').format(SEG_masks[index_init][index]))
                print('Segmentation mask is: {}'.format(SEG_masks[index_init][index]))

            except IndexError:
                print('SEG mask was not available')
                
            print("We got in here because we didn't have any AI files. This could be because no nodules detected by AI or because no files provided by us.")
            # print(colored('Annotated CT is: {}', 'green').format(annotated_CTs_final[index_init][index]))
            # print(colored('Original CT image is: {}', 'green').format(original_CTs_final[index_init][index]))
            print('Annotated CT is: {}'.format(annotated_CTs_final[index_init][index]))
            print('Original CT image is: {}'.format(original_CTs_final[index_init][index]))
            print("\n")
            
    
    print('-----------------------------------------------------------------------------------')
    print("\n")

    #Print all errors that may exist
    if errors_SEG[index_init]!=[] and errors_SEG[index_init]!=['No Segmentation Files available']:
        print('There were errors in the following SEG files (step1): {}'.format(errors_SEG[index_init]))  
    if empty_CT_files[index_init]!=[]:
        print('There were errors in the following CT files (step2): {}'.format(empty_CT_files[index_init]))
    if possible_nodules_excluded[index_init]!=[]:
        print('Possible nodules excluded due to low threshold value (step2): {}'.format(possible_nodules_excluded[index_init]))
    
    try:
        if SEG_masks_errors[index_init]!=[] and (len(SEG_masks_errors[index_init])>1 or (SEG_masks_errors[index_init][0] not in SEG_masks[index_init])): #If more than enough errors in SEG files,
        # or 1 but which not added to SEG files list
            print('Problem with SEG files {}'.format(SEG_masks_errors[index_init]))
    except:
        print("In the Except statement here")
        
    print("Size of segmentation file is {}".format(size_SEG[index_init]))
    print("\n")
 
    end=time.time()
    print("Time it took to run full code is {} secs".format(end-start))
    
    sys.stdout.close()
    sys.stdout = sys.__stdout__  
    
    
    
#Add two new columns to df, one with the IDs of nodules, and one with their slice numbers
IDs=[]
pat_names_ids=[]
for pat,ids in RedCap_ids.items(): #Loop over all IDs and add participants and their IDs to corresponding lists
    pat_names_ids.append(pat)
    IDs.append(ids)

slice_ids=[]
pat_names_slices=[]
for pat,slices in RedCap_pats.items(): #Same as above for participants and their slice numbers
    pat_names_slices.append(pat)
    slice_ids.append(slices)

if list(df_all['participant_id'])==pat_names_ids: #Add IDs as second column (since below we add slices as first again)
    df_all.insert(0,'IDs',IDs)

if list(df_all['participant_id'])==pat_names_slices: #Add slices as first column
    df_all.insert(0,'Slice numbers',slice_ids)
    
    
    
#Taken from stackoverflow.com/questions/54109548/how-to-save-pandas-to-excel-with-different-colors   
#In the final df replace 0 with nan and save it to xlsx file    
writer=pd.ExcelWriter(output_path.split('/')[-2]+'.xlsx',engine='xlsxwriter') #Create df with XlsxWriter as engine

df_all.to_excel(writer,sheet_name='Sheet1',index=False) #Convert dataframe to excel

#Get xlsxwriter workbook and worksheet objects
workbook=writer.book
worksheet=writer.sheets['Sheet1']

#Add a format - Light red fill with dark red text
format1=workbook.add_format({'font_color':'#9C0006','bg_color':'#FFC7CE'})

#Set conditional format range
start_row=1
end_row=len(df_all)
start_col=2 #Start at index 2 since the first to columns are the IDs and the slices with nodules
end_col=3

#Taken from xlsxwriter.readthedocs.io/working_with_conditional_formats.html
#Apply conditions to cell range
worksheet.conditional_format(start_row,start_col,end_row,end_col, #For empty cells the above format
                              {'type':'blanks',
                              'format':format1})

worksheet.conditional_format(start_row,start_col,end_row,end_col, #Same for cells tht need to manually checked 'xxx'
                              {'type':'text', 
                              'criteria':'begins with',
                              'value':'xxx',
                              'format':format1})

#https://xlsxwriter.readthedocs.io/working_with_colors.html#colors
format2=workbook.add_format({'bg_color':'orange'}) #New format with orange for cases with errors only in images - These should also be checked manually

worksheet.conditional_format(start_row,start_col,end_row,end_col,
                              {'type':'text', 
                              'criteria':'begins with',
                              'value':'!!!',
                              'format':format2})

writer.close() #Save the writer - if .save() is used then excel file saved as 'locked'


#Below some failure cases extracted from comments in the code:

#Failure cases when same slice more than two times in GT slices
#TP images are not extracted properly for now
#Might fail if GT slice in original_CTs (not in 'Results') and not any AI nodules
#If two FN in the same slice it might fail
#Probably fails when one TP and one FP in the same slice (as in 673634) 
#Fails when 3 consecutive FP slices