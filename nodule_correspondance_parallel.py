# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:56:31 2021

@author: soyrl
"""

#os.path.join(root,file)
#os.path.splitext(file)
#ff
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
# from collections import Counter

#Input and Output paths
data_path="H:\My Desktop/moderate_scans/" #Folders with scans #moderate_scans/"#noemphysema_scans/"#advanced_scans/"
output_path="H:\My Desktop/moderate_scans_3-8/" #Any name #noemphysema_pat/"#moderate_pat/" #advanced_pat/"

#Path of ground truth nodules for a specific emphysema degree
ground_truth_path="H:\My Desktop/moderate_gt/"#noemphysema_gt/"#moderate_gt/" #advanced_gt/"

#Path of AI nodules and their volume - created with file aorta_calcium_lungnodules.ipynb
AI_path="H:\My Desktop/allemphexper_AI.xlsx"#no_emphysema_AI.xlsx" #no_emphysema #newnoemph #mod_AI

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
    'We also get possible errors. At last the file names of the slices of the AI output scan are returned'
    'It is assumed that the input path contains only DICOM files of a specific patient'
    
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
                                        cv2.imwrite(output_path+'/'+str(image.shape[0])+'_'+str(file[:-4])+'_mask_no_box'+'.png',mask_red_thresh) #Save it
                                        
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
    
    #If no SEG files provided then keep slice number from AI outputs     
    # if size_SEG==[]:
    #     size_SEG=size_AI
    #!!! error in 944714 in mod
    # size_SEG=size_AI
 
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
    slice_errors=[]
    
    SEG_slice_files={} #Dictionary with file names of SEG files and the image of slice with most nodule pixels
            
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
                    
                    slice_errors=[] #To be filled with possible slices in which their name is different from the dicom instance number attribute
                    
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
                                        cv2.imwrite(output_path+'/'+str(file[:-4])+'_SEG_no_box'+'.png',thresh) #Nodules without bounding box around them - values only 0 or 255
                                        
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
                                        
                                        if num_slice in slicenum[-num_maxs:]: #Loop over slices with maximum number of nodule pixels only
                                            cv2.imwrite(output_path+'/'+str(file[:-4])+'_slice_'+str(num_slice)+'_max'+'.png',SEG_image_color)
                                            #Add image name and slice along with image pixels to dictionary
                                            SEG_slice_files[file+'_slice_'+str(num_slice)+'_max_colored']=SEG_image
                                            
                                            # #For a colored-like version - Binary image with box
                                            # plt.savefig(output_path+'/'+str(file[:-4])+'_'+'slice_'+str(num_slice)+'_max_colored'+'.png',dpi=dpi) #Colored nodule
                                          
                                        else: #Also save the slices not with a maximum number of pixels. Some of them may be needed
                                            cv2.imwrite(output_path+'/'+str(file[:-4])+'_slice_'+str(num_slice)+'_not_max'+'.png',SEG_image_color)
                                            
                                            # #For a colored-like version
                                            # plt.savefig(output_path+'/'+str(file[:-4])+'_slice_'+str(num_slice)+'not_max_colored'+'.png',dpi=dpi)
                                            
                                            SEG_slice_files[file+'_slice_'+str(num_slice)+'_not_max_colored']=SEG_image
                                        
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
                                     
                            else: #In case that slice in the CT file name is different than the dicom attribute
                                slice_errors.append(CTfile)
                                                            
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
            return [],original_CTs,annotated_CT_files,[],[],[]
        else:
            sys.stdout.close()
            sys.stdout = sys.__stdout__ 
            return SEG_masks,original_CTs_final,annotated_CTs_final,SEG_slice_files, slice_errors, SEG_masks_errors  
        #!!! SEG_slice_files,slice_errors, not used!!! DELETE
    else: #In case no SEG files exist
        sys.stdout.close()
        sys.stdout = sys.__stdout__ 
        return [],original_CTs,annotated_CT_files,[],[],[]


# def combine_slices(input_list):
#     'Gets a list as input and split it into a list of lists based on consecutive'
#     'elements. For example, a list like [150,151,176,177] will become [[150,151],[176,177]]'
    
#     #Initialize empty lists to be filled below
#     groups_combine=[]
#     remove_elems=[]
    
#     for elem in input_list: #Loop over elements
#         if (elem-1) not in input_list: #If the previous element does not exist then start tracking elements to combine and remove
#             remove_elems=[]
#         while elem in input_list: #loop over consecutive elements
#             remove_elems.append(elem) #add them in a list to remove
#             elem=elem+1 #increase element until it's not in the input list
#         groups_combine.append(list(np.unique(remove_elems))) #add that list of consecutive elements to final list
#         while elem in input_list: #Same procedure but now in case that order is reversed
#             remove_elems.append(elem)
#             elem=elem-1
#         groups_combine.append(list(np.unique(remove_elems)))
    
#     comb_elems=list(np.unique(groups_combine)) #Get unique elements
    
#     merged_list=[] #To be used below
    
#     try: #since we may have an empty list and get error withh 'comb_elems[0]' below
#         if isinstance(comb_elems[0],list)!=True: #If we don't have a list of lists then all elements should be merged and considered as one list (one nodule)
#             for x in groups_combine: #Merge list based on unique occurences
#                 if x not in merged_list:
#                     merged_list.append(x)
#             comb_elems=merged_list  
#     except:
#         pass        
    
#     return comb_elems

                                                
# start=time.time() #To count time to run all steps

#1st function - extract CT, AI files, number of nodules, size of AI, CT and SEG files, and SEG errors
CTfiles, size_SEG, errors_SEG, AI_num_nods, size_CTs,size_AI, AI_slice_names =zip(*Parallel(n_jobs=-1)(delayed(CT_files_extract)(path,outputs[index],dpi=200) for index,path in enumerate(inputs)))

#2nd function - correspondance between annotated and original scans
original_CTs, annotated_CT_files, empty_CT_files, possible_nodules_excluded=zip(*Parallel(n_jobs=-1)(delayed(annotated_CTs_to_normal_scan)(path,CTfiles[index],outputs[index]) for index,path in enumerate(inputs)))

#3rd function - correspondance between SEG files, annotated_CT images, and original scan slices
SEG_masks,original_CTs_final,annotated_CTs_final,SEG_slice_files,slice_errors,SEG_masks_errors=zip(*Parallel(n_jobs=-1)(delayed(mask_seg_slices)(original_CTs[index],path,outputs[index],annotated_CT_files[index],dpi=200) for index,path in enumerate(inputs)))
 

#Output File - Final computations
patient_names=[] #List to be filled with the patient IDs
all_volumes=[] #List to be filled with final list of volumes of nodules (TP and FP)
FP_num=[] #List to be filled with the number of FP findings for each patient

AI_pats={} #AI slices with nodules
AI_pats_vols={} #Volumes of AI slices with nodules
AI_pats_slices={} #All AI slices - original names
RedCap_pats={} #Slices with RedCap annotations
RedCap_pats_vols={} #Volumes of RedCap annotated slices

for index_init,path in enumerate(inputs): #Loop over each patient  

    sys.stdout = open(outputs[index_init]+'output.txt', 'w') #Save output here
    
    patient_names.append(path.split('/')[-1]) #Add patient ID to this list
    
    #!!MAYBE USED LATER?
    #Initialize empty lists to store the volumes of TP and FN findings from REDCap for each patient
    # TP_volumes=[]
    # FN_volumes=[]
    
    #Load ground truth nodules in a dataframe and get an array of integers with the slices of the ground truth nodules
    try: #To ensure that manual annotations from REDCap exist
        ground_truth_nodules=pd.read_csv(ground_truth_path+path.split('/')[-1]+'.csv') #Read CSV file with REDCap annotations
        
        slice_cols=[col for col in ground_truth_nodules.columns if 'slice' in col] #Get list of column names with nodules
        slice_vals=ground_truth_nodules[slice_cols].values #Get list of slices with nodules
        slice_vals=slice_vals[~np.isnan(slice_vals)] #Exclude NaN values
        slice_vals=slice_vals.astype(int) #Convert them to integers
        print("Slices with nodules from REDCap are {}".format(slice_vals))
        RedCap_pats[path.split('/')[-1]]=slice_vals
        
        #!!!!! check first and last 10 slices for assumption below
        for j in slice_vals:
            if j<10 or j>size_CTs[index_init]-11: #between 20-30 slices exist
                print("PROBLEM: In patient {} first 10 or last 10 slices contain nodule - Slice {}".format(path.split('/')[-1],j))
        
        ids=[col for col in ground_truth_nodules.columns if 'nodule_id' in col] #Get list of column names with nodule IDs
        nodule_ids=ground_truth_nodules[ids].values #Get list of nodule IDs
        nodule_ids=nodule_ids[~np.isnan(nodule_ids)] #Exclude NaN values
        nodule_ids=nodule_ids.astype(int) #Convert them to integers
        print("Nodule IDs are: {}".format(nodule_ids))

        vol_ids=[col for col in ground_truth_nodules.columns if 'volume_solid' in col] #Get list of column names with volume of solid nodules
        volume_ids=ground_truth_nodules[vol_ids].values #Get values of volumes of solid nodules
        
        missing=np.where(np.isnan(volume_ids)) #Find where we don't have a value for volumes of solid nodules
        vol_ids_sub=[col for col in ground_truth_nodules.columns if 'volume_subsolid' in col] #Get list of column names with volume of subsolid nodules
        volume_ids_sub=ground_truth_nodules[vol_ids_sub].values #Get values of volumes of subsolid nodules
        volume_ids_sub=volume_ids_sub[missing] #Get values of volumes of subsolid nodules 
        
        volume_ids[np.isnan(volume_ids)]=volume_ids_sub #Replace NaN values in volumes of solid nodules with the volumes of subsolid values
        volume_ids=volume_ids[~np.isnan(volume_ids)] #Remove NaNs
        volume_ids=volume_ids.astype(float) #Convert to float
        print("Volumes of the above nodules are: {}".format(volume_ids))
        print("\n")
        RedCap_pats_vols[path.split('/')[-1]]=volume_ids

    except: #If manual annotations file not found
        print("ERROR: No manual Annotations file from REDCap available")
        slice_vals=[]
        continue


    try:
        df=pd.read_excel(AI_path,index_col=0) #Read Dataframe with AI nodules, using first column as indices
        # AI_nod_num=df.loc[int(path.split('/')[-1])]['num_nodules'] #!!!do i need this?
            
        #!!! ALSO TRY WHICH NUMBER OF AI EG L08 CORRESPONDS TO REDCAP ANNOTS!!
        for file in os.listdir(path): #Loop over all AI files and get information from first slice (0)
            if len(file.split('.')[3])>1 and int(file.split('.')[4])==0:
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
        # print(actual_slice)

        #!!!!! volumes extract from df!!!
        volumes_AI=df.loc[int(path.split('/')[-1])][:10].values[df.loc[int(path.split('/')[-1])][:10].values!='-']
        AI_pats_vols[path.split('/')[-1]]=volumes_AI #Add volumes in AI_pats_vols
        print("Slices with nodules from AI are {}".format(AI_pats[path.split('/')[-1]]))
        print("Their volumes are: {}".format(volumes_AI))

    except: #If any error print it
        print(traceback.format_exc())
        pass

                    
    AI_pats_slices[path.split('/')[-1]]=AI_slice_names[index_init] #Add to AI_pats_slices all AI slice names
    #!!!!was AI_slices2[index_init]

    #Initialize empty lists to be filled below
    tp_final=[]
    fp_final=[]
    fn_final=[]
    
    
    # final_tp=[]
    # final_fp=[]
    rem_FP_may=[] #!!!RENAME THIS LIST
    # tp_final_new=[]
    # tp_AI=[]
    # temp_FP=[] #!!probably delete soon after chacek
    sure_FP=[]
    no_annots=[]
    keep_track=[]
    # no_TP=[]
    # actual_TP=[]
    to_check=[]
    # already_taken=[]
    # test_AI_slice=[]
    error_FP=[]
    
    #!!!! new and shorter path -USED BELOW!
    orig_check_sl=[ctname.split('.')[4] for ctname in original_CTs[index_init]]
    print(orig_check_sl) #maybe diff from RedCap slices with +- a few
    print('\n')
    
    #!!!!sorted for consistency!
    slice_vals=np.sort(slice_vals)#.sort()
    AI_pats[path.split('/')[-1]]=np.sort(AI_pats[path.split('/')[-1]])#.sort()
    
    for GT_slice in slice_vals:
        
        found=0
        possible_TP=[]   
        # ultimate_GT_slice=[]
        tp_final_tentative=[]
        rem_FP_may_tentative=[]
        # tp_AI_tentative=[]
        
        tp_final_correct=[]
        rem_FP_may_correct=[]
        # tp_AI_correct=[]
        
        for indexct,ctname in enumerate(original_CTs[index_init]):#CTfiles:
            
                    
         
            # if ctname.split('.')[4]==int(GT_slice) and len(ctname.split('.')[3])==1:
            #     ultimate_GT_slice.append(GT_slice)
                
            for slice_range in range(GT_slice-5,GT_slice+6):
                #FOr 136470 with FP not getting in the loop below - there are FP not taken into account - this is why below added
                if int(ctname.split('.')[4])==slice_range: # in range(GT_slice-5,GT_slice+6):
                    for slice_AI in AI_pats[path.split('/')[-1]]:
                        for slice_AI_range in range(int(slice_AI)-15,int(slice_AI)+16): 
                            #!!!!changed from - 5 to +6 to adress failure in 591162
                            if int(slice_AI_range)==int(ctname.split('.')[4]):# and len(ctname.split('.')[3])==1: #!!!!! DO I NEED SECOND CONDITION?
                                #int(size_CTs[index_init]-int(slice_AI_range))
                                manual_CT=dicom.dcmread(path+'/'+ctname)#original_scan_slices[index])
                                image_manual_CT=manual_CT.pixel_array #Load CT
                                
                                    
                                manual_annotation=dicom.dcmread(path+'/'+annotated_CT_files[index_init][indexct])#manual_annotations[index])
                                image_manual_annotation=manual_annotation.pixel_array

                                
                                #Find locations of different pixels between original slice and annotations
                                differences_CT_annot=np.where(image_manual_CT!=image_manual_annotation) 
                                # print('differences CT')
                                # print(differences_CT_annot)
                                im_new=np.zeros((512,512))
                               
                                im_new[differences_CT_annot]=image_manual_annotation[differences_CT_annot]
                                #Keep only the region in which nodule exists in manual annotyations
                
                                im_new[410:,468:]=0 #Set to zero bottom right region with 'F'
                                cv2.imwrite(outputs[index_init]+'/'+ctname+'_manual_annotations.png',im_new)
                
                                
                                im_new2=cv2.threshold(im_new,1,255,cv2.THRESH_BINARY)[1]
                               
                                   
                                                                                        # print('Below in loop')
                                # print('GT_slice {}'.format(GT_slice))
                                # print('ctname {}'.format(ctname))
                                # print('slice_range {}'.format(slice_range))
                                # print('slice_AI {}'.format(slice_AI))
                                # print('slice_AI_range {}'.format(slice_AI_range))
                                # # print('AI_slice_num {}'.format(AI_slice_num))
                                # # print(' {}'.format())
                                # print('\n')
                                                        
                                                        
                                for AI_slice_num in AI_pats_slices[path.split('/')[-1]]:
                                    if int(slice_AI_range)==size_AI[index_init]-int(AI_slice_num.split('.')[4]):
                                        
                                        # print("INSIDE!!")
                                        # print('GT_slice {}'.format(GT_slice))
                                        # print('ctname {}'.format(ctname))
                                        # print('slice_range {}'.format(slice_range))
                                        # print('slice_AI {}'.format(slice_AI))
                                        # print('slice_AI_range {}'.format(slice_AI_range))
                                        # # print('AI_slice_num {}'.format(AI_slice_num))
                                        # # print(' {}'.format())
                                        # print('\n')
                                                
                                        AI_dicom=dicom.dcmread(path+'/'+AI_slice_num)
                                        image_AI=AI_dicom.pixel_array #Load CT
                                    
                                                #Resize AI image to (512,512) - same size as SEG and CT files below, convert to HSV and get mask for red and yellow
                                        AI_image=image_AI.copy() #As a good practice - to ensure that we don't change the original image
                                        AI_512=cv2.resize(AI_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC) #Resize to (512,512)
                                        AI_hsv=cv2.cvtColor(AI_512,cv2.COLOR_BGR2HSV) #Convert from BGR to HSV colorspace
                                        mask_im_red=cv2.bitwise_and(AI_hsv,AI_hsv, mask=cv2.inRange(AI_hsv, (100,0,0), (200, 255, 255))) #Red mask - lower and upper HSV values
                                        mask_im_red[0:50,:,:]=0 #Set top pixels that mention 'Not for clinical use' to zero - ignore them and keep only nodules

                                        #Until here mask_im_red is a color image with range of values from 0-255
                                        #Now we convert from BGR (emerged with bitwise operation) to grayscale to have same shape (512,512) as SEG files
                                        #It is used below to take a thresholded image
                                        mask_im_red_gray=cv2.cvtColor(mask_im_red,cv2.COLOR_BGR2GRAY) #Also changes range of values
                                
                                        #Get a smoothed (thresholded) image with only nodules
                                        #!!! WHY NOT 1 INSTEAD OF 128? TRY THAT AND SEE RESULTS!
                                        mask_red_thresh = cv2.threshold(mask_im_red_gray,128,255,cv2.THRESH_BINARY)[1] 
                                        cv2.imwrite(outputs[index_init]+'/'+ctname+'_maskredthresh.png',mask_red_thresh)
                                        # print(np.where(mask_red_thresh!=0))
                                        # print(np.where(mask_red_thresh!=0)!=[])
                                        # print(np.where(mask_red_thresh!=0)[0]!=[])
                                        # print(np.where(mask_red_thresh!=0)[1]==[])
                                        # # print(np.where(mask_red_thresh!=0).size)
                                        # print(np.where(mask_red_thresh!=0)[0].size)
                                        
                                        #ADDRESS FAILURE IN 369762 WHERE TP WHILE FN - there are annotated slices (180)-meets 179 AI that not nodules in RedCap (175)
                                        #ALTHOUGH VERY SPECIFIC! IF slice_range=GT_slice+1 then we would have failure again
                                        #!!!last condition len(np.where) to address failure in 429789 where slice-5 affects correct one!!!!
                                        # if np.where(mask_red_thresh!=0)[0].size==0 and len(np.where(np.array(orig_check_sl)==str(GT_slice))[0])<2:
                                        #     # if int(slice_range)==int(GT_slice):
                                        #         # print(ctname.split('.')[4])
                                        #         # print(np.where(np.array(orig_check_sl)==str(ctname.split('.')[4])))
                                        #         # print(np.where(np.array(orig_check_sl)==str(ctname.split('.')[4]))[0])
                                        #         no_annots.append(GT_slice)
                                        #         #!!!IS IT CERTAINLY FN?
                                        #         test_AI_slice.append(slice_AI)
                                                # #ADDRESS FAILURE IN MOD 985215
                                                # if slice_AI in AI_pats[path.split('/')[-1]]:
                                                #     actual_TP.append(GT_slice)
                                        if np.where(mask_red_thresh!=0)[0].size==0:
                                            #ONLY FOR SPECIFIC CASE WHERE GTSLICE IN ORIGCT! FOR THE REST FAILS!
                                            if int(slice_range)==int(GT_slice):
                                                no_annots.append(GT_slice)
                                        #!!!ADDRESS FAILURE IN MOD 985215
                                        # if np.where(mask_red_thresh!=0)[0].size==0 and len(np.where(np.array(orig_check_sl)==str(GT_slice))[0])>=2:
                                        #     no_TP.append(GT_slice)

                                        if np.where(mask_red_thresh!=0)[0].size>0:#len(np.unique(mask_red_thresh))>1:
                                            differences_AI_annot=np.where(im_new2==mask_red_thresh) 
                                            #!!!! was !=
                                            #above refer to same nodule but with dif contours from AI and manual annot! Not to different nod locations!!
                                            # print('dif AI')
                                            # print(differences_AI_annot)
                                            im_AI=np.zeros((512,512))
                                            im_AI[differences_AI_annot]=mask_red_thresh[differences_AI_annot]
                                            cv2.imwrite(outputs[index_init]+'/'+ctname+str(slice_AI)+'_AI_annotations.png',im_AI)
                                            print("GT_slice is {}".format(GT_slice))
                                            print("slice_range=ctname is {}".format(slice_range))
                                            # print(AI_slice)
                                            print("slice_AI_range is {}".format(slice_AI_range))
                                            print("AI_slice_num is {}".format(AI_slice_num))
                                            print("slice_AI is {}".format(slice_AI))
                                            print('\n')
                                            
                                            # print('tp first {}'.format(tp_final))
                                            # print('fn first {}'.format(fn_final))
                                            # print('fp sure {}'.format(sure_FP))
                                            
                                            # if int(slice_range)==int(GT_slice):
                                                
                                            
                                            
                                            # print("mask thresh")
                                            # print(np.unique(mask_red_thresh))
                                            # print('im_AI')
                                            # print(np.unique(im_AI))
                                            # print("further INSIDE!!")
                                            # print('GT_slice {}'.format(GT_slice))
                                            # print('ctname {}'.format(ctname))
                                            # print('slice_range {}'.format(slice_range))
                                            # print('slice_AI {}'.format(slice_AI))
                                            # print('slice_AI_range {}'.format(slice_AI_range))
                                            # # print('AI_slice_num {}'.format(AI_slice_num))
                                            # # print(' {}'.format())
                                            # print('\n')
                                            
                                            if len(np.unique(im_AI))>1: #true nodule
                                                    # print(np.unique(im_AI))
                                                    # print(np.where(im_AI!=np.unique(im_AI)[0]))
                                                    # print(len(np.where(im_AI!=np.unique(im_AI)[0])[0]))

                                            # if len(differences_AI_annot[0])>1 or len(differences_AI_annot[1])>1:
                                                # if np.sum([1 for i in slice_vals if i==GT_slice])==1: #!!! WHY THIS?
                                                    # print('diff AI annot')
                                                    # print(differences_AI_annot)
                                                    # print('unique im_AI')
                                                    # print(np.unique(im_AI))
                                                    #!!!need to fix condition below with other conditions below
                                                    
                                                    # if GT_slice not in tp_final and int(slice_range)==int(GT_slice):
                                                    #     tp_final_correct.append(GT_slice) ##slice_range) #
                                                    #     rem_FP_may_correct.append(slice_AI)
                                                    #     tp_AI_correct.append(slice_AI)
                                                    #     print('image len >1 and true slide')
                                                    
                                                    
                                                    #!!MAYBE ADDRESS A LOT OF FAILURES LIKE 971099 AND MAYBE OTEHRS
                                                    to_check.append(slice_AI)
                                                    # print('tp here {}'.format(tp_final))
                                                    #!!!ADDRESS FAILURE IN 673634
                                                    if GT_slice not in tp_final and slice_AI not in rem_FP_may and GT_slice not in keep_track: #and int(slice_range)!=int(GT_slice): #problem in 255903 - I don't know why but two times in same GT_slice (294) and sameall the others!
                                                        
                                                        if slice_range in slice_vals and GT_slice!=slice_range: #Failure in 673634
                                                            print('in but out')

                                                        else:
                                                            # print('ADDED CHECK CONFIRM')
                                                            tp_final.append(GT_slice) ##slice_range) #
                                                            rem_FP_may.append(slice_AI)
                                                            # tp_AI.append(slice_AI)
                                                            
                                                            # tp_final_tentative.append(GT_slice) ##slice_range) #
                                                            # rem_FP_may_tentative.append(slice_AI)
                                                            # tp_AI_tentative.append(slice_AI)
                                                            # possible_TP.append(slice_range) #GT_slice
                                                            # print(differences_AI_annot)
                                                            # cv2.imwrite(outputs[index_init]+'/'+ctname+'_manual_annoationsnew.png',im_new2)
                                                            # cv2.imwrite(outputs[index_init]+'/'+ctname+'_mred.png',mask_red_thresh)
                
                
                                                            found=1                                                        # print('Below in loop')
                                                        # print('GT_slice {}'.format(GT_slice))
                                                        # print('ctname {}'.format(ctname))
                                                        # print('slice_range {}'.format(slice_range))
                                                        # print('slice_AI {}'.format(slice_AI))
                                                        # print('slice_AI_range {}'.format(slice_AI_range))
                                                        # print('AI_slice_num {}'.format(AI_slice_num))
                                                        # # print(' {}'.format())
                                                        # print('\n')
                                                    elif GT_slice not in tp_final and slice_AI in rem_FP_may and GT_slice not in keep_track:
                                                        if slice_range in slice_vals and GT_slice!=slice_range: #Failure in 673634
                                                            print('in but out in elif to fix 810826')
                                                            #here also for 985215,591162 - no effect so ok
                                                        else:
                                                            tp_final.append(GT_slice) ##slice_range) #
                                                            rem_FP_may.append(slice_AI)
                                                            # tp_AI.append(slice_AI)
                                                            print('IN ELIF TO FIX 810826 - HOPE ONLY FOR THIS')
                                                            #+163557 but ok
                                                            #!!!!recheck above case!! should it happen again would it be ok?
                                                            error_FP.append(slice_AI)
                                                            found=1

                                                    #!!! error in 670208 and 845594 with 2 same slices in GT
                                                    elif GT_slice in tp_final and len(np.where(np.array(slice_vals)==GT_slice)[0])>1 and slice_AI not in rem_FP_may:
                                                            tp_final.append(GT_slice) ##slice_range) #
                                                            rem_FP_may.append(slice_AI)
                                                            found=1
                                                            print('FOR CASES WITH 2 SAME SLICES IN GT SHOULD GET HERE')
                                                        
                                                    # else:
                                                    #     print("NOT IN THE LOOP THIS TIME")
                                                    #     print('GT_slice {}'.format(GT_slice))
                                                    #     print('slice_AI {}'.format(slice_AI))
                                            
                                            #!!!BELOW TO ADDRESS 136154 IN WHICH THERE ARE TWO DIFFERENT ANNOTATIONS IN 
                                            #!!2 CONSECUTIVE SLICES BUT ONLY THE NON-AI DETECTED NODULE IS CORRECT  
                                            #len(np.unique(im_AI))<=1 and  - not needed
                                            #!!!last condition len(np.where) to address failure in 429789!!!!
                                            elif GT_slice in tp_final and int(slice_range)==int(GT_slice) and len(np.where(np.array(orig_check_sl)==str(GT_slice))[0])<2: #was ctname.split('.')[4]
                                                        # tp_final_correct.append(GT_slice) ##slice_range) #
                                                        ind_remove=np.where(np.array(tp_final)==GT_slice)
                                                        tp_final.remove(GT_slice)
                                                        del rem_FP_may[ind_remove[0][0]]
                                                        # del tp_AI[ind_remove[0][0]]
                                                        # rem_FP_may_correct.append(slice_AI)
                                                        # # tp_AI_correct.append(slice_AI)
                                                        # fn_final.append(GT_slice)
                                                        print('image len <=1 and true slide')
                                                        sure_FP.append(slice_AI)
                                                        fn_final.append(GT_slice)
                                                        
                                                        #!!!ADDRESS FAILURE IN 673634
                                                        keep_track.append(GT_slice)
                                            
                                            # else:#if len(np.unique(im_AI))>1 GT_slice not in :
                                            #     #!!!below if to address failure in 585377 when increase AI range for 591162
                                            #     # if slice_AI in rem_FP_may:
                                            #     #     pass
                                            #     if GT_slice in keep_track:
                                            #         print("Not added {} slice in temp_FP".format(GT_slice))
                                            #         pass
                                                # else:
                                                #     temp_FP.append(slice_AI)
                                                #     #!! catches fp error in 335382
                                                #     print('entered temp_FP')
                                     
                    #!!MAYBE ADDRESS A LOT OF FAILURES LIKE 971099 AND MAYBE OTEHRS
                    if len(to_check)>1:
                        if [x for val_check in to_check for x in range(int(val_check)-15,int(val_check)+15) if x==rem_FP_may[-1]]!=[]: #added to adress extra fp in 673634
                        #above condition added since we may add slices to to_check without actually adding a TP, meaning that we will replace wrong FP slice
                            # print('remfp first {}'.format(rem_FP_may))
                            # print('tpfinal {}'.format(tp_final))
    
                            print('Adaptations due to wrong slice selection for FP')
                            print(to_check)
                            distance=1000
                            kept=-1
                            for ind_sel,select in enumerate(to_check):   
                                if np.abs(GT_slice-select)<distance:
                                    distance=np.abs(GT_slice-select)
                                    keep=ind_sel
                                    
                                # elif np.abs(GT_slice-select)==distance:
                                    
                                    
                            print(to_check[keep])
                            # if np.abs(to_check[keep]-rem_FP_may[-1])>15:
                            #     print('Should only get here to fix error in 673634 - dont know why happened')
                            if to_check[keep] not in rem_FP_may:# and np.abs(to_check[keep]-rem_FP_may[-1])<=15:#already_taken: #rem_FP_may: #!!LINE ADDED TO ADRESS FAILURE IN 985215 AFTER FIXING 971099
                                rem_FP_may[-1]=to_check[keep]
                            # already_taken.append(to_check[keep])
                            
                            # print('remfp second {}'.format(rem_FP_may))
                    to_check=[]
                        
                        
          
                                        # else:

                    # break
        if found==0:
            fn_final.append(GT_slice)   

    # print('remfp third {}'.format(rem_FP_may))
    # print('tpfinal third{}'.format(tp_final))
    #!! TO ADDRESS CASES LIKE 136470 WHERE fp not taken into account/added in the above lists.
    ## for vol_ind,slice_with_FP in enumerate(AI_pats[path.split('/')[-1]]):
    for slice_with_FP in (AI_pats[path.split('/')[-1]]):
        if slice_with_FP not in sure_FP and slice_with_FP not in rem_FP_may:# and slice_with_FP not in temp_FP:
            sure_FP.append(slice_with_FP)
            print('slice {} with FP not added initially'.format(slice_with_FP))
            
        # elif slice_with_FP not in sure_FP and slice_with_FP not in rem_FP_may and slice_with_FP not in temp_FP:
        #     print('!!!!!CHANGE FROM!!!!!!!!!!DONT DELETE temp_FP')
    
    # if temp_FP!=[]:
    #     for slice_with_temp_FP in temp_FP:
    #         if slice_with_temp_FP not in sure_FP and slice_with_temp_FP not in tp_final and slice_with_temp_FP not in rem_FP_may:
    #             sure_FP.append(slice_with_temp_FP)
    #             print('moved temp_FP to sure_FP')
    
    #!!!FAILURE IN 892519 WHERE 2 SAME SLICES OF AI, ONE TP ONE FP
    # added=[]
    # for multiple_slice in rem_FP_may:
    #     if len(np.where(np.array(AI_pats[path.split('/')[-1]])==multiple_slice)[0])>=2 and multiple_slice not in added:
    #         print('Adding same slice as FP')
    #         sure_FP.append(multiple_slice)
    #         added.append(multiple_slice)
    
    #!!!failure in 673634
    #!!! FAILURES OF 892519 WHERE 2 SAME SLICES OF AI, ONE TP ONE FP AND 673634 BOTH FP
    for missing_slice in AI_pats[path.split('/')[-1]]:#rem_FP_may:
        counting=0
        total_AI=len(np.where(np.array(AI_pats[path.split('/')[-1]])==missing_slice)[0])
        if missing_slice in rem_FP_may: #meaning that also exists in TP
            counting=counting+len(np.where(rem_FP_may==missing_slice)[0])+len(np.where(sure_FP==missing_slice)[0])
            if counting<total_AI:
                print("slice in TP fixed")
                for i in range(total_AI-counting):
                    sure_FP.append(missing_slice)
        else: #either exist in FP or not at all
            counting=counting+len(np.where(sure_FP==missing_slice)[0])
            if counting<total_AI:
                print('slice in FP fixed')
                for i in range(total_AI-counting):
                    sure_FP.append(missing_slice)
                    
        counting=0
    
    
    #ADRESS FAILURE IN 810826
    
    for GT_slice in fn_final:
        if GT_slice in tp_final:
            fn_final.remove(GT_slice)
            print("CHANGE FROM fN TO TP - HOPE ONLY FOR 810826")
    
    #ADDRESS FAILURE IN 369762 WHERE TP WHILE FN
    # print('tp before noannot {}'.format(tp_final))
    # print('fn before noannot {}'.format(fn_final))
    # print('fp before noannot {}'.format(sure_FP))
    # print('no_annots before noannot{}'.format(no_annots))
    # print('remfp before noannot {}'.format(rem_FP_may))
    for wrong_annot in no_annots:
        if wrong_annot in tp_final:# and wrong_annot not in actual_TP: #!!SECOND CONDIITON TO ADDRESS erro in MOD 985215
        ##!!! SECOND CONDITION RESULTS IN WRONG RESULT IN 369762
            ind_remove=np.where(np.array(tp_final)==wrong_annot)
            tp_final.remove(wrong_annot)
            #ADRESS FAILURE IN 810826
            if rem_FP_may[ind_remove[0][0]] not in error_FP:
                sure_FP.append(rem_FP_may[ind_remove[0][0]])
            else:
                print('CHANGE FROM HOPE IN HERE ONLY FOR 810826')
                
            del rem_FP_may[ind_remove[0][0]]
            # del tp_AI[ind_remove[0][0]]
            # rem_FP_may_correct.append(slice_AI)
            # # tp_AI_correct.append(slice_AI)
            # fn_final.append(GT_slice)
            print('no_annotations issues fixed')
            fn_final.append(wrong_annot)
      
    #!!!FOR ERRROR IN 985215
    # for wrong_TP in no_TP:
    #     if wrong_TP in tp_final:
    #         ind_remove=np.where(np.array(tp_final)==wrong_TP)
    #         tp_final.remove(wrong_TP)
    #         sure_FP.append(rem_FP_may[ind_remove[0][0]])
    #         del rem_FP_may[ind_remove[0][0]]
    #         del tp_AI[ind_remove[0][0]]
    #         # rem_FP_may_correct.append(slice_AI)
    #         # # tp_AI_correct.append(slice_AI)
    #         # fn_final.append(GT_slice)
    #         print('no_TP issues fixed')
    #         fn_final.append(wrong_TP)
            
    #!!!! NOT SURE IF CORRECT!!! drress failure in 673634 where tp should be fn   
    # delete_rem_FP_may=[]#!!! not done yet
    # for indx,prob_FP in enumerate(rem_FP_may):
    #     if prob_FP in sure_FP:
    #         delete_rem_FP_may.append(prob_FP)
    #         print('will delete from rem_FP_may and change TP  {} to FN'.format(tp_final[indx]))
    #         fn_final.append(tp_final[indx])
    #         del tp_final[indx]
    
    try:
        assert len(tp_final)+len(fn_final)==len(slice_vals)
        assert len(tp_final)+len(sure_FP)==len(AI_pats[path.split('/')[-1]])
    
    except:
        print("ERROR!! CHECK FILE AGAIN!!!")
        print(traceback.format_exc())
        pass

    print('tp first {}'.format(tp_final))
    print('fn first {}'.format(fn_final))
    print('fp sure {}'.format(sure_FP))
    print('\n')
    
    # print('test_AI_slice {}'.format(test_AI_slice))
    print('no_annots {}'.format(no_annots))
    # print('actual_TP {}'.format(actual_TP))

    # print('tempfp {}'.format(temp_FP))
    print('remfp first {}'.format(rem_FP_may))
    if len(rem_FP_may)!=len(tp_final):
        print("LIST rem_fp_may DOES NOT CONTAIN ONLY TP MATCHING WITH AI")
    
    # print('notp {}'.format(no_TP))

    print('\n')
    
    
    
    
    
    
    
    
    
    
    
# #Double occurence in 810826 
# #!!! check whgat happens if more than two times!!!
#     counting=dict(Counter(rem_FP_may)) #[rem_FP_may.count(rem_FP_may[i]) for i in range(len(rem_FP_may))]
#     for double_elem in list(counting.keys()):
#         if counting[double_elem]==2:#>1:
#             AI_occurences=list(AI_pats[path.split('/')[-1]]).count(double_elem)
            
#             if AI_occurences<=1:
#                     indices_occur=[ind for ind,val in enumerate(rem_FP_may) if val==double_elem]
#                     duplicates_FP=[rem_FP_may[i] for i in indices_occur]
#                     duplicates_TP=[tp_final[i] for i in indices_occur]
#                     keep=np.argmin([np.abs(i-j) for i,j in zip(duplicates_FP,duplicates_TP)])
#                     for i in indices_occur:
#                         if i!=keep:
#                             tp_final_new.append(i)
#                             # tp_AI.append(i)
#             else:
#                 print("MAYBE PROBLEM WITH RESULTS! CHECK AGAIN MANUALLY!")
    
#     # for ind in tp_final_new:
#     #     tp_final.remove(tp_final[ind])
#     tp_final_new2=[]
#     # if tp_final_new!=[]:
#     for ind in tp_final_new:
#         tp_final_new2.append(tp_final[ind])
#         fn_final.append(tp_final[ind])
#         # tp_AI.append(rem_FP_may[ind])
#     # else:
      
#     # for elem in rem_FP_may: #136581 wouldn't otherwise work for Ai slices
#     #     if elem not in tp_AI:
#     #         tp_AI.append(elem)
    
#     if tp_final_new2!=[]:
#         print('tpnew {}'.format(tp_final_new))
#         print('tp in removing {}'.format(tp_final))
#         print('tp final new2 {}'.format(tp_final_new2))
#         tp_final=list(set(tp_final)-set(tp_final_new2))  
#         if len(tp_AI)!=len(np.unique(tp_AI)):
#             print("Duplicate AI slices in TP")
#         # tp_AI=np.unique(tp_AI) #changes order and messed everything
#         if len(tp_final)!=len(tp_AI):
#             print('ERROR! CHECK MANUALLY!!')
#     # print('tp_final2 {}'.format(tp_final))
            

#         # minimum=20
#         # list_min=[]
#         # for ind,val in enumerate(tp_final):#possible_TP):
#         #       check=np.abs(val-GT_slice)   
#         #       if check<minimum:
#         #           minimum=check
#         #           list_min.append(ind)
                 
#         # for elem in list_min:
#         #     final_tp.append(tp_final[elem]) #possible_TP


#     for slice_AI in AI_pats[path.split('/')[-1]]:     
#         if (slice_AI not in tp_final) and (slice_AI not in fn_final) and (slice_AI not in final_tp):
#             fp_final.append(slice_AI)
                
#     print("FP final new are :{}".format(fp_final))

#     for fp in rem_FP_may:
#         try:
#             fp_final.remove(fp)
#         except:
#             pass
            
#     # print('possible_TP {}'.format(possible_TP))
#     # print("TP final new are :{}".format(tp_final))
#     print("remov FP maybe new are :{}".format(rem_FP_may))
#     print('\n')
#     print('ACTUAL RESULTS BELOW')
#     print('tp_final2 {}'.format(tp_final))
#     print("Their lesion number are {}".format([nodule_ids[ind2] for ind,val in enumerate(tp_final) for ind2,val2 in enumerate(slice_vals) if val==val2]))
#     print("their volumes are {}".format([volume_ids[ind2] for ind,val in enumerate(tp_final) for ind2,val2 in enumerate(slice_vals) if val==val2]))
    
#      # print("final_tp new are :{}".format(final_tp))
#     print("Tp AI slices are {}".format(tp_AI))
#     print("Their lesion number are {}".format([ind+1 for ind,val in enumerate(AI_pats[path.split('/')[-1]]) if val in tp_AI]))
#     print("Their volumes are {}".format([volumes_AI[ind] for ind,val in enumerate(AI_pats[path.split('/')[-1]]) if val in tp_AI]))

#     print('\n')
#     print("FP final new are :{}".format(fp_final))
#     print("Their lesion number are {}".format([ind+1 for ind,val in enumerate(AI_pats[path.split('/')[-1]]) if val in fp_final]))
#     print("Their volumes are {}".format([volumes_AI[ind] for ind,val in enumerate(AI_pats[path.split('/')[-1]]) if val in fp_final]))

#     print('\n')
#     print("FN final new are :{}".format(fn_final))
#     print("Their lesion number are {}".format([nodule_ids[ind2] for ind,val in enumerate(fn_final) for ind2,val2 in enumerate(slice_vals) if val==val2]))
#     print("Their volumes are {}".format([volume_ids[ind2] for ind,val in enumerate(fn_final) for ind2,val2 in enumerate(slice_vals) if val==val2]))
#                                          #for ind,val in enumerate(slice_vals) if val in fn_final]))


   
#     print('-----------------------------------------------------------------------------------')
#     print("\n")













    slice_vals_found=[] #Empty lists to be filled with the slice values from REDCap that were detected by AI below
    
    #Initialize empty lists to be filled with slices with TP, FP and FN based on REDCap annotations
    slices_TP=[]
    slices_FP=[]
    slices_FN=[]
        
    #Get which detections of AI correspond to which slices of the original CT scan
    #There are also FP detections from AI that are not taken into account here
    detectionsCT=[]
    detectionsAI=[]
#!!!!!
# problem here!! - usually empty both the lists below!!
    # if AI_pats[path.split('/')[-1]]!=[]:
    #     for AI_file in AI_pats[path.split('/')[-1]]:
    if list(AI_num_nods[index_init].keys())!=[]: #If there are slices with nodules in AI outputs
        for AI_file in AI_num_nods[index_init].keys(): #Loop over AI slices with nodules
            for orig_CT_slice in original_CTs[index_init]: #Loop over original CT slices with nodules
                ###!!!!!!size_SEG instead of size_CTs
                #!!!!AI_file.split('.')[4]
                if int(AI_file.split('.')[4])==size_CTs[index_init]-int(orig_CT_slice.split('.')[4]): #If there is correspondance between slices of original CT and AI output (reverse order) then we have a match
                    detectionsCT.append(orig_CT_slice) #Add original CT slice to list
                    detectionsAI.append(AI_file) #Add AI output slice to list
                    
    else: #If there are no AI detections
        if slice_vals.size!=0: #And if there are manual annotations
            for i in slice_vals: #All slices in manual annotations should be considered FNs
                slices_FN.append(i) #Add that to FN slices
            print('IMPORTANT!: There no AI detections but there are manual annotations')
        else:
            print("IMPORTANT!: There are no AI detections and there are no manual annotations")
            print("We should get 0TP, 0TN, and 0FP")
#!!!!!!!!
    print(detectionsAI)
    print(detectionsCT)

    FPcheck=[] #List of original scan slices for which we don't have a manual annotation but nodules detected by AI
    #Could also be TP since for a nodule we only have one slice in manual annotations while it may be extended in more.

    if size_AI[index_init]!=size_CTs[index_init]:
        print("ERROR: Original CT scan does not have same number of slices as AI output")
        print('\n')
        
        #!!!!!delete!!
    if int(size_AI[index_init])<int(size_CTs[index_init]): #!!!!-40 to address CT files that may be annot files
            print("ERROR: There may be missing AI Slices!") #!!! changed to 40 from 50 due to error in 944714 in mod
            print('\n') #!!!! 'may be' since in 670208 we have many annot ~85

    # if AI_pats[path.split('/')[-1]]!=[]:
    #     for AInod in AI_pats[path.split('/')[-1]]:
    if list(AI_num_nods[index_init].keys())!=[]: #If we have AI detections
        for AInod in AI_num_nods[index_init].keys(): #Loop over the detected slices with nodules
            if AInod not in detectionsAI: #Nodules for which no correspondance with annotated images
            #!!!
            # ###!!!!!!size_SEG instead of size_CTs
                FPcheck.append(size_CTs[index_init]-int(AInod.split('.')[4])) #!!!.split('.')[4]
        # print("FPcheck is {}".format(FPcheck))
        
        #If above lists not with unique elements, then some AI slices exist at least 2 times!
        # if len(np.unique(FPcheck))!=len(FPcheck):
        #     print("Remark:Some AI slices exist at least 2 times!")
               
        for j in np.unique(FPcheck):
            slices_FP.append(j) #Add unique FPcheck slices to FP list (take care of possible TP here below)
                          
        #Get all slice numbers for which we have AI detections
        AI_slices=[int(AI_file.split('.')[4]) for AI_file in AI_num_nods[index_init].keys()] 
        # AI_slices=[int(AI_file) for AI_file in AI_pats[path.split('/')[-1]]] 

        #Get slices for which we have correspondance
        detectionAI_slices=[int(detection_AI_file.split('.')[4]) for detection_AI_file in detectionsAI]
        # detectionAI_slices=[int(detection_AI_file) for detection_AI_file in detectionsAI]

        
        for index,AI_file_slice in enumerate(AI_slices): #Loop over all AI detections
            sum_slice=0 #Initialize a counter
            for slice_no in range(AI_file_slice-10,AI_file_slice+11): #Check current AI slice and 10 before and 10 after it
                #!!! ASSUMED THE FOLLOWING: The bigger the above + - the greater the chances not to have error!
                #10+- seems nice since highly likely that nodule slice is in the middle of nodule slices(where most pixs belong to nod)
                #Check 10 slices before and 10 after - First and last 10 not important since difficult to find nodule there
                if slice_no in detectionAI_slices: #If we have a manual annotation 
                    sum_slice=sum_slice+1
            
            if sum_slice==0: #If we don't have any manual annotations in -+10 slices of the current one, then it's a FP               
                ###!!!!size_SEG instead of size_CTs
                # if size_CTs[index_init]-int(list(AI_pats[path.split('/')[-1]])[index]) not in slices_FP: #If this slice not already in FPs
                #     slices_FP.append(size_CTs[index_init]-int(list(AI_pats[path.split('/')[-1]])[index])) 
                if size_CTs[index_init]-int(list(AI_num_nods[index_init].keys())[index].split('.')[4]) not in slices_FP: #If this slice not already in FPs
                    slices_FP.append(size_CTs[index_init]-int(list(AI_num_nods[index_init].keys())[index].split('.')[4])) 
            
            
    #If we have AI outputs, print the correspondance for all (SEG_files, AI outputs, Original CT slices and Annotated CT slices)
    # if list(AI_num_nods[index_init].keys())!=[]: 
    if list(AI_pats[path.split('/')[-1]])!=[]: 

        
        for ind1,CT in enumerate(detectionsCT): #Loop over original CT slices for which we have AI detections
            for ind2, orig_CT in enumerate(original_CTs_final[index_init]): #Loop over a similar list of original slices for which we have AI detections but with a different order (index is used below so that there is correspondance)
                
                if CT==orig_CT: #If we have the same CT slice in both lists

                    if int(original_CTs_final[index_init][ind2].split('.')[4]) in slice_vals: #Check if this is an actual nodule
                    #!!!!same as orig_CT.split('.')[4]
                            try:
                                for i in np.where(slice_vals==int(original_CTs_final[index_init][ind2].split('.')[4]))[0]: #Loop over indices of REDCap slices that exist in annotated CTs as well
                                    nod_ids=int(nodule_ids[i]) #Get IDs of these nodules from REDCap (L01 etc. in Syngo.via manual annotations)
                                    volumes=float(volume_ids[i]) #Get volumes of these nodules
                                    print("Ground truth: The volume of this is {}, the ID in manual annotations is {}, and the slice is {}".format(volumes,nod_ids,np.unique(slice_vals[np.where(slice_vals==int(original_CTs_final[index_init][ind2].split('.')[4]))])))
                                    # tp_vols.append(volumes) #!!!! delete or keep?
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
                            
                            #slices_TP may not be unique if we have double AI slices - export them twice by mistake
                            slices_TP.append(int(original_CTs_final[index_init][ind2].split('.')[4])) #Add that slice to a list
                    else: #possible FP - maybe also be TP that extends from the slice that exists in slice_vals
                        for i in slice_vals: #Check slices close to slice_vals to see if we have annotations - maybe also be TP but for now added in FP
                            if int(original_CTs_final[index_init][ind2].split('.')[4]) in range(i-1,i-5,-1) or int(original_CTs_final[index_init][ind2].split('.')[4]) in range(i+1,i+6):
                                print("High chances of having TP (even though not same slice as in REDCap or no annotation file available) for the following:")
                                try:
                                    nod_ids=int(nodule_ids[np.where(slice_vals==int(original_CTs_final[index_init][ind2].split('.')[4]))])
                                    volumes=float(volume_ids[np.where(slice_vals==int(original_CTs_final[index_init][ind2].split('.')[4]))])
                                    print("If so, the volume of this is {} and the ID in manual annotations is {} and slice is {}".format(volumes,nod_ids,slice_vals[np.where(slice_vals==int(original_CTs_final[index_init][ind2].split('.')[4]))]))
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
                        
                        #If the above slice not already in FPs added there for now
                        if int(original_CTs_final[index_init][ind2].split('.')[4]) not in slices_FP:
                                slices_FP.append(int(original_CTs_final[index_init][ind2].split('.')[4])) #Add that slice to the FP list

         
        for ind3, CT_file in enumerate(original_CTs_final[index_init]): #Print files with nodules not found by the AI
            if CT_file not in detectionsCT: #Since it contains only slices that were also found by the AI
                if int(original_CTs_final[index_init][ind3].split('.')[4]) in slice_vals: #Check if this is an actual nodule
                    slice_vals_found.append(int(original_CTs_final[index_init][ind3].split('.')[4])) #To make sure that we have taken it into account
        #!!!! i am here            
                    # try:
                    #     nod_ids=int(nodule_ids[np.where(slice_vals==int(original_CTs_final[index_init][ind2].split('.')[4]))])
                    #     volumes=float(volume_ids[np.where(slice_vals==int(original_CTs_final[index_init][ind2].split('.')[4]))])
                    #     print("The volume of this is {} and the ID in manual annotations is {}".format(volumes,nod_ids))
                    #     print('second if')
                    # except:
                    #     pass
                    
                    print('True nodule not detected by AI:')
                    try:
                        # print(colored('Segmentation mask is: {}', 'green').format(SEG_masks[index_init][ind3]))
                        print('Segmentation mask is: {}'.format(SEG_masks[index_init][ind3]))

                    except IndexError:
                        pass
                        # print('no SEG mask')
                    # print(colored('Annotated CT is: {}', 'green').format(annotated_CTs_final[index_init][ind3]))
                    # print(colored('Original CT image is: {}', 'green').format(original_CTs_final[index_init][ind3]))
                    print('Annotated CT is: {}'.format(annotated_CTs_final[index_init][ind3]))
                    print('Original CT image is: {}'.format(original_CTs_final[index_init][ind3]))
                    # print("The volume of this is {} and the ID in manual annotations is {}".format(volumes,nod_ids))

                    print("\n")
                    slices_FN.append(int(original_CTs_final[index_init][ind3].split('.')[4]))
                else:
                    print("No true nodule and not detected by AI or no annotation file available")
                    try:                 
                        # print(colored('Segmentation mask is: {}', 'green').format(SEG_masks[index_init][ind3]))
                        print('Segmentation mask is: {}'.format(SEG_masks[index_init][ind3]))

                    except:
                        pass
                        # print('no SEG mask')
                    
                    # print(colored('Annotated CT is: {}', 'green').format(annotated_CTs_final[index_init][ind3]))
                    # print(colored('Original CT image is: {}', 'green').format(original_CTs_final[index_init][ind3]))
                    print('Annotated CT is: {}'.format(annotated_CTs_final[index_init][ind3]))
                    print('Original CT image is: {}'.format(original_CTs_final[index_init][ind3]))
                    print("\n")
         
        #KEEP FOR WHEN ONLY SEG FILES AS BASELINE!!
        # for ind4, CT_file_rest in enumerate(original_CTs[index_init]): #Print files with nodules found by the AI and not having SEG file
        #     if CT_file_rest not in original_CTs_final[index_init] and CT_file_rest in detectionsCT:
        #         for ind5,detection_CT in enumerate(detectionsCT):
        #             if detection_CT==CT_file_rest:
        #                 print(colored('Annotated CT is: {}', 'green').format(annotated_CT_files[index_init][ind4]))
        #                 print(colored('Original CT image is: {}', 'green').format(original_CTs[index_init][ind4]))
        #                 print(colored('AI output image with nodules is: {}', 'green').format(detectionsAI[ind5]))
        #                 print("\n")
        #                 FP=FP+1
        #                 slices_FP.append(int(original_CTs_final[index_init][ind4].split('.')[4]))
           
        for ind4, CT_file_rest in enumerate(original_CTs[index_init]): #Print files with nodules not found by the AI and not having SEG file
        #!!!! JUST ADDED CT_file_rest in slice_vals
            if CT_file_rest not in detectionsCT and CT_file_rest in slice_vals: #CT_file_rest not in original_CTs_final[index_init] and 
                print('CHECK IF WE SHOULD BE HERE!!!!!!') #!!!!!!!!!!
                # print(colored('Annotated CT is: {}', 'green').format(annotated_CT_files[index_init][ind4]))
                # print(colored('Original CT image is: {}', 'green').format(original_CTs[index_init][ind4]))
                print('Annotated CT is: {}'.format(annotated_CT_files[index_init][ind4]))
                print('Original CT image is: {}'.format(original_CTs[index_init][ind4]))
                print("\n")
                slices_FN.append(int(original_CTs[index_init][ind4].split('.')[4])) #CTs_final
    
    #We don't expect to get in the else loop below 
    else: #otherwise print only SEG files, annotated CT images and original CT slices (if no AI outputs) - only for SEG files that exist
        # print("There are no AI outputs, only manual segmentations")
        # print("NO AI OUTPUTS!!! ONLY ANNO FILES/SEG")
        for index in range(len(original_CTs_final[index_init])): 
            try:
                # print(colored('Segmentation mask is: {}', 'green').format(SEG_masks[index_init][index]))
                print('Segmentation mask is: {}'.format(SEG_masks[index_init][index]))

            except IndexError:
                print('noSEG mask')
            # print(colored('Annotated CT is: {}', 'green').format(annotated_CTs_final[index_init][index]))
            # print(colored('Original CT image is: {}', 'green').format(original_CTs_final[index_init][index]))
            print('Annotated CT is: {}'.format(annotated_CTs_final[index_init][index]))
            print('Original CT image is: {}'.format(original_CTs_final[index_init][index]))
            print("\n")
            
    
    print('-----------------------------------------------------------------------------------')
    print("\n")

    #Print all errors that may exist
    if errors_SEG[index_init]!=[]:
        print('There were errors in the following SEG files (step1): {}'.format(errors_SEG[index_init]))  
    if empty_CT_files[index_init]!=[]:
        print('There were errors in the following CT files (step2): {}'.format(empty_CT_files[index_init]))
    if possible_nodules_excluded[index_init]!=[]:
        print('Possible nodules excluded due to low threshold value (step2): {}'.format(possible_nodules_excluded[index_init]))
    if SEG_masks_errors[index_init]!=[] and (len(SEG_masks_errors[index_init])>1 or (SEG_masks_errors[index_init][0] not in SEG_masks[index_init])): #If more than enough errors in SEG files,
    # or 1 but which not added to SEG files list
        print('Problem with SEG files {}'.format(SEG_masks_errors[index_init]))
        
    print("Size of segmentation file is {}".format(size_SEG[index_init]))
    print("Size of AI output scan is {}".format(size_AI[index_init]))
    print("Size of original CT scan is {}".format(size_CTs[index_init]))
    print("\n")

    if size_CTs[index_init]!=size_AI[index_init]:
        print('ERROR: There are either missing AI slices or original CT scan slices')
    
    
    # try:
    #     slice_vals.size==0 #==[]:
    #     # print("No Ground Truth annotations available")
    # except:
    #     try:
    #         slice_vals==[]
    #         # print("No Ground Truth annotations available")
    #     except:
    #         pass
    
        # pass
    
    # try:
    #     slice_vals==[]
    #     print("No annotation file available")
    # except:
    #     pass
        

    # print('Actual number of ground truth nodules is {}'.format(len(slice_vals)))
    
    # print("BEFORE Slices with TP are: {}".format(np.unique(slices_TP)))
    print("BEFORE Slices with TP are: {}".format(np.unique(slices_TP)))

    print("BEFORE Slices with FP are: {}".format(np.unique(slices_FP)))
    print("BEFORE Slices with FN are: {}".format(np.unique(slices_FN)))
    
    # if list(AI_num_nods[index_init].keys())==[]: #AI_nodules
    #     print('NO AI outputs!!')
        #!!!DEL?
 
    end=time.time()
    print("Time it took to run full code is {} secs".format(end-start))
    
    sys.stdout.close()
    sys.stdout = sys.__stdout__  


sys.stdout = open(output_path+'final_results.txt', 'w') #Save output here
# print("Patient names are {}".format(patient_names))
# print("Volumes of the above patients for TP and FN are {}".format(all_volumes))
# print("Number of FP for each of the above patients is {}".format(FP_num))

# print("Total number of patients is {}".format(len(patient_names)))
# print("Total number of volume information is {}".format(len(all_volumes)))
# print("Total number of FP patients is {}".format(len(FP_num)))


sys.stdout.close()
sys.stdout = sys.__stdout__  







    
 
    
 
    
 
#!!!!!! JUST COMMENTED EVERYTHING BELOW AFTER NEW IMPLEMENTAION    
#     try:
#         slice_vals.size!=0 #Make sure that we have manually annotated nodules
#         remove_FN_FP_slices=[]
#         #We may not have annotations for the exact slices in slice_vals and that's why we may miss some slices with FNs
#         possible_FNs=list((((set(slice_vals)-set(slice_vals_found))))) #Nodules in manual annotations not found above - may be FN
        
#         # print("slices_FP here are {}".format(slices_FP))
#         # print("possible FNs are {}".format(possible_FNs))
#         # print('slice_vals {}'.format(slice_vals))
#         # print('slice_vals_found {}'.format(slice_vals_found))
        
#         for pos_FN in possible_FNs:
#             if pos_FN in slices_FP: #If a nodule in manual annotations and considered as FP
#             #(since AI detects it but no annotation file for this slice but only maybe for close ones)
#             #then it should actually be TP. So, delete from FP list and add to TP
#             #!!!! NOT ALSO CHANGE TP ETC NUMBER OF NODS? LIKE TP=TP+1
#                 slices_TP.append(pos_FN)
#                 remove_FN_FP_slices.append(pos_FN)
#                 # print(pos_FN)
        
#         for elem in remove_FN_FP_slices:
#             try:
#                 while True: #MAYBE MORE THAN ONE TIMES SINCE MAYBE DOUBLE ai SLICES
#                     slices_FP.remove(elem)
#                     # print(elem)
#             except:
#                 pass
        
#         for elem in remove_FN_FP_slices:
#             try:
#                 while True: #Since we may have the same slice twice in slice_vals and so, in possible_FNs
#                     possible_FNs.remove(elem)
#             except:
#                 pass
            
#         # num_remove=0    
#         # for elem in possible_FNs:
#         #     if elem in slices_FN:
#         #          # print('Slices with FN common elements with FN slices not found in slice_vals')
#         #          num_remove+=1
#         #          print("elem in this loop!!! {}".format(elem))
                 
#         # print("remove_FN_FP_slices are {}".format(remove_FN_FP_slices))    
#         # print("Slices with TP are prelast: {}".format(slices_TP))
#         # for key,value in same_slice.items():
#         #     slices_TP.append(key)
#         #     TP=TP+value-1
#         # print("Slices with TP before unique are: {}".format(slices_TP))

#         #For cases in which we have the same nodule more than 1 times in manual annotations (because of doubel AI slices)
#         # print("NON UNIQUE SLICES_TP ARE {}".format(slices_TP))
#         slices_TP=np.unique(slices_TP)
#         slices_TP_new=list(slices_TP)
#         for i in slices_TP:
#             if list(slice_vals).count(i)>1:
#                 for j in range(list(slice_vals).count(i)-1):
#                     slices_TP_new.append(i) #add this slices additional times to TP list
                    
#         slices_TP=slices_TP_new
#         # print("a {}".format(slices_TP))
        
        
#         # print("Slices with TP are: {}".format(slices_TP))
#         # print("Slices with FP are: {}".format(np.unique(slices_FP)))
#         # print("Slices with FN are: {}".format(np.unique(slices_FN)))
        
#         # print('FN - slices from slice_vals not found are: {}'.format(possible_FNs))
#         # #!!!np.unique in TP slices below!!!!!!!
#         # print('We have {} TP, {} FP, and {} FN'.format(len((slices_TP)),
#         #len(np.unique(slices_FP)),len(np.unique(slices_FN))+len(np.unique(possible_FNs))-num_remove))#TP,FP,FN))  #+len(list(set(slice_vals)-set(slice_vals_found)))
#         # #!!!np.unique in TP slices below!!!!!!!
#         # print("Total num of TP,FP is {}".format(len((slices_TP))+len(np.unique(slices_FP))))
#         # print('Total AI nodules found: {}'.format(len(np.unique([num.split('.')[4] for num in list(AI_nodules[index_init].keys())]))))
#     except: #was else
#             #!!!np.unique in TP slices below!!!!!!!
#         # print('We have {} TP, {} FP, and {} FN'.format(len((slices_TP)),len(np.unique(slices_FP)),len(np.unique(slices_FN))))#TP,FP,FN))  
#         pass

#     # if list(AI_num_nods[index_init].keys())==[]: #AI_nodules
#     #     print("MISSING AI OUTPUT or no AI detections!!!!!!!")
#     print("AFTER1 Slices with TP are: {}".format(np.unique(slices_TP)))

#     print("AFTER1 Slices with FP are: {}".format(np.unique(slices_FP)))
#     print("AFTER1 Slices with FN are: {}".format(np.unique(slices_FN)))
# #!!!! confirm same nodule as in slice_vals detected by Ai and not in other locations on same slice
#     tp_and_fp=[]
#     fp_slice=[]
#     tp_slice=[]
#     confirmed_tp_fp=[]
#     # print('There are {} AI slices with nodules'.format(len(list(AI_num_nods[index_init].keys()))))#AI_annotations)))
#     print('There are {} AI slices with nodules'.format(len(list(AI_pats[path.split('/')[-1]]))))#AI_annotations)))

    
#     # for slice_or in 
#     #!!!!!!
#     # AI_files_only=[]
#     AI_pats_slices[path.split('/')[-1]]=[]
#     for file_AI in os.listdir(path):
#         if len(file_AI.split('.')[3])>1:
            
#             dicom_file=dicom.dcmread(path+'/'+file_AI) #Read file
#             img=dicom_file.pixel_array
#             if img.shape[0]==1024:
#                 if np.sum(img[900:,960:])!=0: #Ensure that we don't have graphs with HU information
#                     AI_pats_slices[path.split('/')[-1]].append(file_AI)

#     # btrbtr
#     #Loop over slices that contain nodules
#     #There may or may not be in slice_vals since we may have annotations of +- a few slices
#     # for index,filename in enumerate(list(AI_num_nods[index_init].keys())):#AI_annotations):
#     for index,filename in enumerate(list(AI_pats_slices[path.split('/')[-1]])):#AI_annotations):
# #!!!!seems we can use it here!!
#         #Loop over slices for which we have annotations
#         for indexct,ctname in enumerate(original_CTs[index_init]):#CTfiles:
            
#             #Ensure that we have original ct slice and not manual annotations
#             ####!!!!size_SEG instead of size_CTs
#             if str(size_CTs[index_init]-int(filename.split('.')[4])) in ctname.split('.')[4] and len(ctname.split('.')[3])==1:
#                 # print('we are in and ai {} orig {}'.format(filename,ctname))
#         # if int(original_scan_slices[index].split('.')[4]) in slices_TP:
#         #size_SEG[index_init]-int(original_scan_slices[index].split('.')[4])==filename.split('.')[4]:
                
#                 manual_CT=dicom.dcmread(path+'/'+ctname)#original_scan_slices[index])
#                 image_manual_CT=manual_CT.pixel_array #Load CT
#                 # plt.figure()
#                 # plt.imshow(image_manual_CT) 
#                 # plt.show()
#                 # image_manual_CT = cv2.threshold(image_manual_CT,128,255,cv2.THRESH_BINARY)[1] 
    
#                 # cv2.imshow('or',image_manual_CT)
#                 # cv2.waitKey(0)
#                 # plt.ioff()
#                 # plt.figure()
#                 # plt.imshow(image_manual_CT) 
#                 # plt.title('file')
#                 # plt.savefig(output_path+'/'+'_'+str(filename[:-4])+'_or.png',dpi=300) # file[:-4] was used to avoid '.dcm' ending
#                 # plt.close()
                
#                 # for indexan, anname in enumerate(annotated_CT_files[index_init]):
#                 #     if anname==ctname:
#                 #         (annotated_CT_files[index_init][ind4]))
#                 # print(colored('Original CT image is: {}', 'green').format(original_CTs[index_init][ind4]))
                    
#                 manual_annotation=dicom.dcmread(path+'/'+annotated_CT_files[index_init][indexct])#manual_annotations[index])
#                 image_manual_annotation=manual_annotation.pixel_array
#                 # cv2.imshow('or2',image_manual_annotation)
#                 # cv2.waitKey(0)
#                 # plt.ioff()
#                 # plt.figure()
#                 # plt.imshow(image_manual_annotation) 
#                 # plt.title('file')
#                 # plt.savefig(output_path+'/'+'_'+str(filename[:-4])+'_an.png',dpi=300) # file[:-4] was used to avoid '.dcm' ending
#                 # plt.close()
                
#                 #Find locations of different pixels between original slice and annotatiosn
#                 differences_CT_annot=np.where(image_manual_CT!=image_manual_annotation) 
                    
#                 im_new=np.zeros((512,512))
               
#                 im_new[differences_CT_annot]=image_manual_annotation[differences_CT_annot]
#                 #Keep only the region in whihc nodule exists in manual annotyations

#                 im_new[410:,468:]=0 #Set to zero bottom right region with 'F'
#                 cv2.imwrite(outputs[index_init]+'/'+ctname+'_manual_annoations.png',im_new)

                

                
#                 # plt.ioff()
#                 # plt.figure()
#                 # plt.imshow(im_new)
               
#                 # plt.title('file')
#                 # plt.savefig(output_path+'/'+'_'+str(filename[:-4])+'_dif.png',dpi=300) # file[:-4] was used to avoid '.dcm' ending
#                 # plt.close()
                
#                 # cv2.imwrite(output_path+'/'+'_'+str(filename[:-4])+'_dif'+'.png',image_manual_annotation[differences_CT_annot[0],differences_CT_annot[1]]) #Save it
                                      
                
#                 AI_dicom=dicom.dcmread(path+'/'+filename)
#                 image_AI=AI_dicom.pixel_array #Load CT
                
#                 # if len(np.where(image_AI[:,:,1]!=image_AI[:,:,2])[0])>600: #threshold for at least 1 nodule - 1000 for at least 2, if needed      
                                          
#                         #Resize AI image to (512,512) - same size as SEG and CT files below, convert to HSV and get mask for red and yellow
#                 AI_image=image_AI.copy() #As a good practice - to ensure that we don't change the original image
#                 AI_512=cv2.resize(AI_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC) #Resize to (512,512)
#                 AI_hsv=cv2.cvtColor(AI_512,cv2.COLOR_BGR2HSV) #Convert from BGR to HSV colorspace
#                 mask_im_red=cv2.bitwise_and(AI_hsv,AI_hsv, mask=cv2.inRange(AI_hsv, (100,0,0), (200, 255, 255))) #Red mask - lower and upper HSV values
#                 mask_im_red[0:50,:,:]=0 #Set top pixels that mention 'Not for clinical use' to zero - ignore them and keep only nodules
                
#                 # Maybe needed for nodule identification - avoid duplicates - also (80,0,70) and (80,0,130) but best (80,100,0)
#                 # mask_im_yellow=cv2.bitwise_and(AI_hsv,AI_hsv, mask=cv2.inRange(AI_hsv, (80,0,110), (110, 255, 255))) #Yellow mask
#                 # cv2.imwrite(output_path+'/'+str(image.shape[0])+'_'+str(file[:-4])+'_yellow'+'.png',mask_im_yellow) #Save yellow mask - range from 0-255
        
#                 #Until here mask_im_red is a color image with range of values from 0-255
#                 #Now we convert from BGR (emerged with bitwise operation) to grayscale to have same shape (512,512) as SEG files
#                 #It is used below to take a thresholded image
#                 mask_im_red_gray=cv2.cvtColor(mask_im_red,cv2.COLOR_BGR2GRAY) #Also changes range of values
                                                               
#                 # if len(np.unique(mask_im_red_gray))!=1: #If there is a nodule in the image
#                     # AI_nodules[file]=mask_im_red_gray #Add file name and grayscale image to a dictionary
        
#                     #Get a smoothed (thresholded) image with only nodules
#                 mask_red_thresh = cv2.threshold(mask_im_red_gray,128,255,cv2.THRESH_BINARY)[1] 
#                 # cv2.imwrite(outputs[index_init]+'/'+ctname+'_maskredbelow.png',mask_red_thresh)
#                 #above same as mask_no_box

            
            
#                 #!!!
#                 # print("y AI before {}".format((np.unique(np.where(mask_red_thresh!=0)[0]))))
#                 # print("x AI before {}".format((np.unique(np.where(mask_red_thresh!=0)[1]))))
#                 ycoords_AI=combine_slices(np.unique(np.where(mask_red_thresh!=0)[0])) #!!!ASSUMING NOT TOUCHING NODULES    
#                 xcoords_AI=combine_slices(np.unique(np.where(mask_red_thresh!=0)[1]))
#                 # print("AFTER COMBINE {}".format(filename))
                
#                 # print(ycoords_AI)
#                 # print(xcoords_AI)
#                 #DEBUG
#                 # if int(ctname.split('.')[4])==362:
#                 #     print(xcoords_AI)
      
#                 #Convert to list
#                 try:
#                     if isinstance(ycoords_AI[0],list)!=True:
#                         ycoords_AI=[ycoords_AI]
#                     if isinstance(xcoords_AI[0],list)!=True:
#                         xcoords_AI=[xcoords_AI]    
#                 except:
#                     pass
                
#                 # print("after listing elements")
#                 # print(ycoords_AI)
#                 # print(xcoords_AI)
#                 #!!!!!ALSO SET CONDITION THAT TWO CONSECUTIVE ELEMENTS (LAST OF LIST I AND FIRST OF LIST I+1)
#                 #SHOULD HAVE AT LEAST A DIFFERENCE OF 10 PIXELS TO BE CONSIDERED DIFFERENT?
#                 #WANT TO ADRESSS CASE OF 1 NODULE WHICH MAY SPLIT IN HALF
        
#                 # try:
#                 #     ycoords_AI==[] #was!=
#                 # except:
#                 #     if isinstance(ycoords_AI[0],list):
#                 #         pass
#                 #     else:
#                 #         ycoords_AI=[ycoords_AI]
#                 #         xcoords_AI=[xcoords_AI]
                
                
#                 im_new2=cv2.threshold(im_new,1,255,cv2.THRESH_BINARY)[1]
#                 ycoord_an=combine_slices(np.unique(np.where(im_new2!=0)[0]))
#                 xcoord_an=combine_slices(np.unique(np.where(im_new2!=0)[1]))
                
#                 try:
#                     if isinstance(ycoord_an[0],list)!=True:
#                         ycoord_an=[ycoord_an]
#                     if isinstance(xcoord_an[0],list)!=True:
#                         xcoord_an=[xcoord_an]   

#                     # if isinstance(ycoord_an[0],list):
#                     #     pass
#                     # else:
#                     #     ycoord_an=[ycoord_an]
#                     #     xcoord_an=[xcoord_an]  
#                 except:
#                     pass
                
#                 # print("after listing elements manual")
#                 # print(ycoord_an)
#                 # print(xcoord_an)
                
#                 # try:
#                 #     ycoord_an==[] #was!=
#                 # except:
#                 #     if isinstance(ycoord_an[0],list):
#                 #         pass
#                 #     else:
#                 #         ycoord_an=[ycoord_an]
#                 #         xcoord_an=[xcoord_an]
            
                
#                 if len(xcoords_AI)!=len(ycoords_AI) or len(xcoord_an)!=len(ycoord_an):
#                     print('UNEXPECTED ERROR IN SLICE {}'.format(int(ctname.split('.')[4])))#original_scan_slices[index].split('.')[4])))
#                     print('x_AI {}, y_AI {}, x_man {}, y_man {}'.format(xcoords_AI,ycoords_AI,xcoord_an,ycoord_an))
#                     ####!!!!size_SEG instead of size_CTs
#                     print("Into loop since {}".format(str(size_CTs[index_init]-int(filename.split('.')[4]))))
#                     print("AI filename {}".format(filename))
                    
#                     #fix error in x ccords of AI
#                     if len(xcoords_AI)>len(ycoords_AI): #FIX 136470,345599 NO EMPHY 
#                         checkelem=0
#                         xcoords_AI_new=[]
#                         for ind,coordx in enumerate(xcoords_AI):
#                             if checkelem+1==coordx[0]-1:
#                                 checkelem=coordx[-1]
#                                 del xcoords_AI_new[-1]#=xcoords_AI_new[:]
#                                 xcoords_AI_new.append(xcoords_AI[ind-1]+[coordx[0]-1]+coordx)
#                             else:
#                                 checkelem=coordx[-1]
#                                 xcoords_AI_new.append(coordx)
#                         xcoords_AI=xcoords_AI_new
#                         if len(xcoords_AI)==len(ycoords_AI):
#                             print("ERROR FIXED in xcoords AI!")
#                             if len(xcoords_AI)==len(xcoord_an) and len(ycoords_AI)==len(ycoord_an):
#                     # print("Slice {} has same num of annotations as nods detected by AI".format(int(original_scan_slices[index].split('.')[4])))
#                                 tp_slice.append(int(ctname.split('.')[4]))
#                         else:
#                             print("ERROR REMAIN in xcoords AI!!!")
#                             if len(xcoords_AI)>len(xcoord_an) or len(ycoords_AI)>len(ycoord_an):
#                                 if len(xcoords_AI)==1 and len(ycoords_AI)==1:
#                                     print("slice {} considered as FP".format(int(ctname.split('.')[4])))
#                                     fp_slice.append(int(ctname.split('.')[4]))#int(original_scan_slices[index].split('.')[4]))
#                                 else:
#                                     print("slice {} considered as TP_FP".format(int(ctname.split('.')[4])))
#                                     tp_and_fp.append(int(ctname.split('.')[4]))#
#                                     confirmed_tp_fp.append(int(ctname.split('.')[4]))

                            
                                
#                     #Fix error in x coords of annot file
#                     if len(xcoord_an)>len(ycoord_an): #FIX 136470,345599 NO EMPHY 
#                         checkelem=0
#                         xcoord_an_new=[]
#                         for ind,coordx in enumerate(xcoord_an):
#                             if checkelem+1==coordx[0]-1:
#                                 checkelem=coordx[-1]
#                                 del xcoord_an_new[-1]#=xcoords_AI_new[:]
#                                 xcoord_an_new.append(xcoord_an[ind-1]+[coordx[0]-1]+coordx)
#                             else:
#                                 checkelem=coordx[-1]
#                                 xcoord_an_new.append(coordx)
#                         xcoord_an=xcoord_an_new
#                         if len(xcoord_an)==len(ycoord_an):
#                             print("ERROR FIXED! in xcoords an")
#                             if len(xcoords_AI)==len(xcoord_an) and len(ycoords_AI)==len(ycoord_an):
#                     # print("Slice {} has same num of annotations as nods detected by AI".format(int(original_scan_slices[index].split('.')[4])))
#                                 tp_slice.append(int(ctname.split('.')[4]))
#                         else:
#                             print("ERROR REMAIN!!! in xcoords an")
#                             if len(xcoords_AI)>len(xcoord_an) or len(ycoords_AI)>len(ycoord_an):
#                                 if len(xcoords_AI)==1 and len(ycoords_AI)==1:
#                                     print("slice {} considered as FP".format(int(ctname.split('.')[4])))
#                                     fp_slice.append(int(ctname.split('.')[4]))#int(original_scan_slices[index].split('.')[4]))
#                                 else:
#                                     print("slice {} considered as TP_FP".format(int(ctname.split('.')[4])))
#                                     tp_and_fp.append(int(ctname.split('.')[4]))#
                            
#                     #fix error in y ccords of AI
#                     if len(ycoords_AI)>len(xcoords_AI): #FIX 136470,345599 NO EMPHY 
#                         checkelem=0
#                         ycoords_AI_new=[]
#                         for ind,coordy in enumerate(ycoords_AI):
#                             if checkelem+1==coordy[0]-1:
#                                 checkelem=coordy[-1]
#                                 del ycoords_AI_new[-1]#=xcoords_AI_new[:]
#                                 ycoords_AI_new.append(ycoords_AI[ind-1]+[coordy[0]-1]+coordy)
#                             else:
#                                 checkelem=coordy[-1]
#                                 ycoords_AI_new.append(coordy)
#                         ycoords_AI=ycoords_AI_new
#                         if len(ycoords_AI)==len(xcoords_AI):
#                             print("ERROR FIXED! in ycoords AI")
#                             if len(xcoords_AI)==len(xcoord_an) and len(ycoords_AI)==len(ycoord_an):
#                     # print("Slice {} has same num of annotations as nods detected by AI".format(int(original_scan_slices[index].split('.')[4])))
#                                 tp_slice.append(int(ctname.split('.')[4]))
#                         else:
#                             print("ERROR REMAIN!!! in ycoords AI")
#                             if len(xcoords_AI)>len(xcoord_an) or len(ycoords_AI)>len(ycoord_an):
#                                 if len(xcoords_AI)==1 and len(ycoords_AI)==1:
#                                     print("slice {} considered as FP".format(int(ctname.split('.')[4])))
#                                     fp_slice.append(int(ctname.split('.')[4]))#int(original_scan_slices[index].split('.')[4]))
#                                 else:
#                                     print("slice {} considered as TP_FP".format(int(ctname.split('.')[4])))
#                                     tp_and_fp.append(int(ctname.split('.')[4]))#
                                
#                     #Fix error in y coords of annot file
#                     if len(ycoord_an)>len(xcoord_an): #FIX 136470,345599 NO EMPHY 
#                         checkelem=0
#                         ycoord_an_new=[]
#                         for ind,coordy in enumerate(ycoord_an):
#                             if checkelem+1==coordy[0]-1:
#                                 checkelem=coordy[-1]
#                                 del ycoord_an_new[-1]#=xcoords_AI_new[:]
#                                 ycoord_an_new.append(ycoord_an[ind-1]+[coordy[0]-1]+coordy)
#                             else:
#                                 checkelem=coordy[-1]
#                                 ycoord_an_new.append(coordy)
#                         ycoord_an=ycoord_an_new
#                         if len(ycoord_an)==len(xcoord_an):
#                             print("ERROR FIXED! in ycoords an")
#                             if len(xcoords_AI)==len(xcoord_an) and len(ycoords_AI)==len(ycoord_an):
#                     # print("Slice {} has same num of annotations as nods detected by AI".format(int(original_scan_slices[index].split('.')[4])))
#                                 tp_slice.append(int(ctname.split('.')[4]))
#                         else:
#                             print("ERROR REMAIN!!! in ycoords an")
#                             if len(xcoords_AI)>len(xcoord_an) or len(ycoords_AI)>len(ycoord_an):
#                                 if len(xcoords_AI)==1 and len(ycoords_AI)==1:
#                                     print("slice {} considered as FP".format(int(ctname.split('.')[4])))
#                                     fp_slice.append(int(ctname.split('.')[4]))#int(original_scan_slices[index].split('.')[4]))
#                                 else:
#                                     print("slice {} considered as TP_FP".format(int(ctname.split('.')[4])))
#                                     tp_and_fp.append(int(ctname.split('.')[4]))#
                   
                            
                   
                    
                   
                    
#                 if len(xcoords_AI)==len(xcoord_an) and len(ycoords_AI)==len(ycoord_an):
#                     # print("Slice {} has same num of annotations as nods detected by AI".format(int(original_scan_slices[index].split('.')[4])))
#                     tp_slice.append(int(ctname.split('.')[4]))#(original_scan_slices[index].split('.')[4]))
#                 # else:
#                 #     print('here for TP')
#                 #     print('len(xcoords_AI) {} len(xcoord_an) {} len(ycoords_AI) {} len(ycoord_an) {}'.format(len(xcoords_AI),len(xcoord_an),len(ycoords_AI),len(ycoord_an)))
#                 #     print('above for slice {}'.format(int(original_scan_slices[index].split('.')[4])))
                    
#                 if len(xcoords_AI)>len(xcoord_an) or len(ycoords_AI)>len(ycoord_an):
#                     if len(xcoords_AI)==1 and len(ycoords_AI)==1:
#                         fp_slice.append(int(ctname.split('.')[4]))#int(original_scan_slices[index].split('.')[4]))
#                     else:
#                         tp_and_fp.append(int(ctname.split('.')[4]))#original_scan_slices[index].split('.')[4]))                
#                 # else:
#                 #     print('len(xcoords_AI) {} len(xcoord_an) {} len(ycoords_AI) {} len(ycoord_an) {}'.format(len(xcoords_AI),len(xcoord_an),len(ycoords_AI),len(ycoord_an)))
#                 #     print('above for slice {}'.format(int(ctname.split('.')[4])))#(original_scan_slices[index].split('.')[4])))
        
        
        
        
#             # common=np.sum(im_new[np.where(mask_red_thresh!=0)]!=0)
#             # print('len common is {}'.format((common)))
#             # if common<5:
#             #     print("DIFFERENT NODULE THAN WHAT FOUND IN {}".format(filename))
      
#     # print('nonuniqueFP slices (for further use) from manual check are {}'.format((fp_slice)))
#     # print('nonuniqueTP slices (for further use) from manual check are {}'.format((tp_slice)))
#     # print('nonuniquePossible slices that might contain both TP and FP are {}'.format((tp_and_fp)))        
#     #!!!EXPECTED ONLY WHEN DUPLICATE AI OR ANNOTATION FILES
#     if len(np.unique(fp_slice))!=len(fp_slice):
#         print("non unique Fp slice {}".format(fp_slice))
#     if len(np.unique(tp_slice))!=len(tp_slice):
#         print("non unique tp slice {}".format(tp_slice)) #136470
#     if len(np.unique(tp_and_fp))!=len(tp_and_fp):
#         print("non unique tp_and_Fp slice {}".format(tp_and_fp))
#     #unique because we may have them more than once due to duplicate Ai files or more than 1 annot files (like in 670208 in advance)
#     print('FP slices (for further use based on annotated_CTs - not REDCap) from manual check are {}'.format(np.unique(fp_slice)))
#     print('TP slices (for further use based on annotated_CTs - not REDCap) from manual check are {}'.format(np.unique(tp_slice)))
#     print('Possible slices that might contain both TP and FP are {}'.format(np.unique(tp_and_fp)))
#     #!!!! USE ABOVE TP IF CORRECT TO COMPARE WITH 'HIGH CHANCES OF TP' - NOT SAME SLICE IN VALS AS IN MANUAL ANNOTS!
#     #!!!THE ABOVE MAY NOT BE THE SAME WITH FINAL LIST SINCE WE USUALLY HAVE MORE ANNOT FILES THAN WHAT IN VALS

#     #!!! OR JUST LOOP IN AI_NUM_NODS KEYS() AND SUBSTRACT FROM IT SET OF FOUND VALUES ABOVE??
#     no_annots=[]
#     # for index,filename in enumerate(list(AI_num_nods[index_init].keys())):#AI_annotations):

#     #     ###!!!!size_SEG instead of size_CTs everywhere below
#     #     if int(size_CTs[index_init]-int(filename.split('.')[4])) not in tp_slice:
#     #         if int(size_CTs[index_init]-int(filename.split('.')[4])) not in fp_slice:
#     #             if int(size_CTs[index_init]-int(filename.split('.')[4])) not in tp_and_fp:
#     #                 # print('no annotations for file {} corresponding to original slice {}'.format(filename,int(size_SEG[index_init]-int(filename.split('.')[4]))))
#     #                 no_annots.append(int(size_CTs[index_init]-int(filename.split('.')[4])))
   
#     for index,filename in enumerate(AI_pats[path.split('/')[-1]]):#AI_annotations):
#         ###!!!!size_SEG instead of size_CTs everywhere below
#         if int(size_CTs[index_init]-int(filename)) not in tp_slice:
#             if int(size_CTs[index_init]-int(filename)) not in fp_slice:
#                 if int(size_CTs[index_init]-int(filename)) not in tp_and_fp:
#                     # print('no annotations for file {} corresponding to original slice {}'.format(filename,int(size_SEG[index_init]-int(filename.split('.')[4]))))
#                     no_annots.append(int(size_CTs[index_init]-int(filename)))              
    
    
#     no_annots.sort()            
#     #!!!!COMPARE THOSE WITH THOSE IN EXCEL FILE MANUALLY CREATED BY ME
#     #TOMAKE SURE THAT ALL DETECTED
#     # print('Unique non-annotated slices of those found by AI are {}'.format(np.unique(no_annots)))
#     if len(np.unique(no_annots))!=len(no_annots): #WHEN DUPLICATE AI OR ANNOT FILES
#         print("non unique non-annotated slices of those found by AI are {}".format((no_annots)))
    
#             # print(mask_red_thresh[differences_CT_annot])
#             # plt.ioff()
#             # plt.figure()
#             # plt.imshow(mask_red_thresh[differences_CT_annot]) 
#             # plt.title('file')
#             # plt.savefig(output_path+'/'+'_'+str(filename[:-4])+'_common.png',dpi=300) # file[:-4] was used to avoid '.dcm' ending
#             # plt.close()
#             # cv2.imwrite(output_path+'/'+'_'+str(filename[:-4])+'_common'+'.png',mask_red_thresh[differences_CT_annot]) #Save it

        
#                     # cv2.imwrite(output_path+'/'+str(image.shape[0])+'_'+str(file[:-4])+'_mask_no_box'+'.png',mask_red_thresh) #Save it
                    
#                     #Get contours for each nodule and create a rectangle around them
#                     # mask_im_red_RGB=cv2.cvtColor(mask_im_red_gray,cv2.COLOR_GRAY2RGB) #Convert grayscale to RGB (not BGR as before)
#                     # contours = cv2.findContours(mask_red_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                     # contours = contours[0] if len(contours) == 2 else contours[1]
#                     # AI_num_nods[file]=len(contours) #!!!!!!
        
        


# #fp=[110, 182, 183, 184, 186, 187, 188, 189, 190, 191, 192, 243, 261, 263, 264, 265, 266, 267, 268, 269, 288, 289, 291, 292, 293, 294, 
# #314, 315, 316, 317, 319, 321, 322, 323, 325, 326, 327, 328, 358, 359, 360, 362, 363, 364, 365, 366, 379, 380, 381, 382, 383, 384, 385, 386]
# #tp=[185, 262, 290, 318, 320, 324, 361, 361]
# #!!!!
#     # TP_FP_modify=[]
#     # TP_notdelete=[]
#     # FP_combine=[]
#     # TP_many=[]
#     # TP_FP_many=[]
#     # FP_combine_many=[]
#     # many_nods=[]
#     # mod_not_del=[]
#     # mod_FP_TP=[]
#     # possible_missed_FP_slices=[]
#     # FP_nods=[]
#     if AI_num_nods[index_init]!={}:
#         for slice_no,num_nods in AI_num_nods[index_init].items():
#             if num_nods==0: #!! ALWAYS TRUE???
#                 print('TERRIBLE ERROR (no nodules) on slice')
#                 ###!!!!size_SEG instead of size_CTs
#                 print(size_CTs[index_init]-int(slice_no.split('.')[4]))
#             # elif num_nods>1
#             #     occurences=np.sum([x.count(37) for x in sli])
#             #     jy
                
#             # else:
#             #     if num_nods>1: #!!! FOR MISSED FP IN 670208
#             #         possible_missed_FP_slices.append(int(slice_no.split('.')[4]))
#             #         FP_nods.append(num_nods)
#                 #!!NOT NEEDED??
#                 # orig_slice=size_SEG[index_init]-int(slice_no.split('.')[4])
#                 # if num_nods>1:
#                 #     TP_FP_many.append(orig_slice)
        
#         # for num_slice in TP_FP_many:
#         #     mod_not_del.append(num_slice)
#         #     for i in range(num_slice-1,num_slice-21,-1):
#         #         if ((i in fp) or (i in tp)) and ((i+1) in mod_not_del):
#         #             mod_not_del.append(i)
#         #             mod_FP_TP.append(i)
#         #     for i in range(num_slice+1,num_slice+21,1):
#         #         if ((i in fp) or (i in tp)) and ((i-1) in mod_not_del):
#         #             mod_not_del.append(i)
#         #             mod_FP_TP.append(i)
#         #     for elem in mod_FP_TP:
#         #         if elem in fp:
#         #             fp.remove(elem)
#         #         if elem in tp:
#         #             tp.remove(elem)
       
            
#             # modif=[num_slice]
#             # for i in range(num_slice-1,num_slice-21,-1):
#             #     if ((i in fp) or (i in tp)) and (i+1) in modif:
#             #         if i not in modif:
#             #             modif.append(i)
                        
#             # for j in modif:      
#             #         many_nods.append(i)
        
#             #     # try:
#             #     #     fp.remove(i)
#             #     #     many_nods.append(i)
#             #     # except:
#             #     #     tp.remove(i)
#             #     #     many_nods.append(i)

#             # modif=[num_slice]
#             # for i in range(num_slice+1,num_slice+21,1):
#             #     if ((i in fp) or (i in tp)) and (i-1) in modif:
#             #         if i not in modif:
#             #             modif.append(i)
                        
#             # for j in modif:    
#             #         many_nods.append(i)
                                        
#             #     # try:
#             #     #     fp.remove(i)
#             #     #     many_nods.append(i)
#             #     # except:
#             #     #     tp.remove(i)
#             #     #     many_nods.append(i)
                    
                    
                    
                    
                    

                
#         # for slice_no,num_nods in AI_num_nods[index_init].items():
#         #     if num_nods==1:
#         #         orig_slice=size_SEG[index_init]-int(slice_no.split('.')[4])
#         #         if orig_slice in TP_FP_many:
                    
#         #             for i in range(orig_slice,orig_slice-21,-1):
#         #                 if (i in TP_FP_many) and ((i in fp) or (i in tp)):
#         #                     try:
#         #                         fp.remove(i)
#         #                     except:
#         #                         tp.remove(i)
                                
#         #             for i in range(orig_slice,orig_slice+21,1):
#         #                 if (i in TP_FP_many) and ((i in fp) or (i in tp)):
#         #                     try:
#         #                         fp.remove(i)
#         #                     except:
#         #                         tp.remove(i)
                    
                    

                        

                

                
                
                
                
#         #         if num_nods==1:
#         #             if orig_slice in slices_TP:
#         #                 TP_notdelete.append(orig_slice) #SLICE SHOULD BE KEPT
#         #                 #!! ASSUME NOD APPEARS 20 BEFORE AND 20 AFTER THIS SLICE
#         #                 for i in range(orig_slice-1,orig_slice-21,-1):
#         #                     if (i in slices_FP) and ((i+1) in TP_notdelete):
#         #                         TP_FP_modify.append(i) #DELETE THAT FROM FP AND ADD TO TP
#         #                         TP_notdelete.append(i) #TO CONTINUE THE LOOP
#         #                 for i in range(orig_slice+1,orig_slice+21,1):
#         #                     if (i in slices_FP) and ((i-1) in TP_notdelete):
#         #                         TP_FP_modify.append(i) #DELETE THAT FROM FP AND ADD TO TP
#         #                         TP_notdelete.append(i) #TO CONTINUE THE LOOP
#         #                 for elem in TP_FP_modify:
#         #                     if elem in slices_FP:
#         #                         slices_FP.remove(elem)
#         #                     if elem not in slices_TP:
#         #                         slices_TP.append(elem)
                                
#         #         elif num_nods>1:
#         #             TP_FP_many.append(orig_slice)
                                
#         # for slice_no,num_nods in AI_num_nods[index_init].items():
#         #     orig_slice=size_SEG[index_init]-int(slice_no.split('.')[4])
#         #     if num_nods==1:
#         #         if orig_slice in slices_FP:
#         #             if (orig_slice+1) in slices_FP or (orig_slice-1) in slices_FP:
#         #                 FP_combine.append(orig_slice)
                        
#     # FP_combine.sort()
#     slices_TP.sort()
#     slices_FN.sort()
#     slices_FP.sort()
#     # print("slices Tp bfore combine {}".format(slices_TP))
#     # print("slices fn bfore combine {}".format(slices_FN))
#     # print("slices fp bfore combine {}".format(slices_FP))
    
#     # slices_FP=combine_slices(FP_combine)
#     #LISTING ONLY IF MANY SEPERATE LISTS. IF INDIVIDUAL SLICES WITH NODULES ONLY, THEN ONLY A SINGLE LIST AT THE END!
#     slices_FP=combine_slices(slices_FP)
#     print("FP slices AFTER COMBI {}".format(slices_FP))
#     print("TP slices at the same point {}".format(slices_TP))
#     # slices_FN=combine_slices(slices_FN) #!!!! i deactivated it!!! MAYBE REACITVE?? 811041 MOD => 2 CONSECUTIVE LICES BUT WITH DIF NODULE!
#     # print("slices fp after combine FOR MISS FN IN 136581{}".format(slices_FP))
#     # print("slices FN after combine {}".format(slices_FN))

    
#     try:
#         if isinstance(slices_FP[0],list):
#             pass
#         else:
#             print("slices fp after combine {}".format(slices_FP))
#             # print("slices FN after combine {}".format(slices_FN))
#             slices_FP=[slices_FP]
#             print("slices FP after listing {}".format(slices_FP))

#     except:
#         pass
    
#     # print("slices FP after listing {}".format(slices_FP))
#         # if slices_FP==[]:
#         #     pass
#         # else:
#         #     slices_FP=[slices_FP]
        
#     # try:
#     #     if isinstance(slices_TP[0],list):
#     #         pass
#     #     else:
#     #         slices_TP=[slices_TP]
#     # except:
#     #     pass
#     # print("SLICESTP FOR CHECK OF MISSING FN IN 136581 ARE {}".format(slices_TP))
#     print('\n')
    
#     # for key,val in AI_num_nods[index_init].items():
#     #     np.sum([x.count(37) for x in sli])
#     #     if int(key.split('.')[4])>1
    
#     #MOVE FPS TO TPS
#     slices_FP_new=[]
#     slices_TP_new=[]
#     # print(slices_FP)
#     for i in range(len(slices_FP)):        
#         try:
#             if slices_FP[i][-1]+1 in slices_TP or slices_FP[i][0]-1 in slices_TP: #If we have the previous or next slice in TPs
#                 # print('normal')
#                 # print(slices_FP)
                
#                 #List of lists is slices_FP. checking last element of list i and first element of list i+1
#                 if slices_FP[i][-1]+1 in slices_TP and slices_FP[i+1][0]-1 in slices_TP:                    
#                     new_nod=slices_FP[i]+[slices_FP[i][-1]+1]+slices_FP[i+1] #merge two lists into new one
#                     print(new_nod)
#                     if len(new_nod)>25:#CASE 591162 NOEMPHYSEMA
#                         new_nod=[slices_FP[i][-1]+1]+slices_FP[i+1]
#                         slices_FP_new.append(slices_FP[i])
#                         print('same error as in 591162')
                    
#                     #670208 where two nodules on same slice in slice_vals
#                     occur1=slices_TP.count(slices_FP[i][-1]+1)
#                     occur2=slices_TP.count(slices_FP[i+1][0]-1)
#                     maxoccur=max(occur1,occur2)
#                     for i in range(maxoccur):                                            
#                         slices_TP_new.append(new_nod)
#                         # print("a newnod {}".format(new_nod))
#                     # if 
                    
#                 if slices_FP[i][-1]+1 in slices_TP and slices_FP[i+1][0]-1 not in slices_TP:
#                     print("into first if!")
#                     new_nod=slices_FP[i]+[slices_FP[i][-1]+1]
                    
#                     for i in range(slices_TP.count(slices_FP[i][-1]+1)):                                            
#                         slices_TP_new.append(new_nod)
#                         # print("b newnod {}".format(new_nod))

#                 try: #!! since fialure to add TP from FP list and also delted from FP list in 383275 noemp
#                     if slices_FP[i+1][0]-1 in slices_TP and slices_FP[i][-1]+1 not in slices_TP:
#                         print("into second if!")
#                         new_nod=[slices_FP[i+1][0]-1]+slices_FP[i+1]
#                         slices_TP_new.append(new_nod)
#                         # print("c newnod {}".format(new_nod))
#                         for i in range(slices_TP.count(slices_FP[i+1][0]-1)):                                            
#                             slices_TP_new.append(new_nod)
#                 except:
#                     if slices_FP[i]==slices_FP[-1] and slices_FP[i][-1]+1 not in slices_TP and slices_FP[i][0]-1 in slice_vals: #! since fialure to add TP from FP list and also delted from FP list in 383275 noemp
#                         new_nod=slices_FP[i]
#                         slices_TP_new.append(new_nod)
#                         print("IN LOOP FOR 383275! SHOULD NOT EXIST ELSEWHERE")
#                         # print("c newnod {}".format(new_nod))
#                         # for i in range(slices_TP.count(slices_FP[i+1][0]-1)):                                            
#                         #     slices_TP_new.append(new_nod)
#                     # pass
                    
#                 if i==0: #!!!GIVES ERROR FOR 136581 IF FOR LOOP USED!!!!! WHY DO WE HAVE IT???
#                     if slices_FP[i][0]-1 in slices_TP:
#                         print("into third if!")
#                         new_nod=[slices_FP[i][0]-1]+slices_FP[i]
#                         # for i in range(slices_TP.count(slices_FP[i][0]-1)):                                            
#                         #     slices_TP_new.append(new_nod)
#                         #     print("d newnod {}".format(new_nod))
                        
#                         print(new_nod)
#                         # print(type(new_nod))
#                         print(slices_TP)
#                         # print(type(slices_TP))
#                         if new_nod not in slices_TP_new: #added for 136581, otherwise appears 3 times!
#                             print("ADDED!!")
#                             slices_TP_new.append(new_nod)
                
                


#             elif slices_FP[i][-1]+1 in slice_vals:
#                 # for ind_num,hit in enumerate(slice_vals):
#                 #     if 
#                     print("INTO FIRST ELIF")
#                     new_nod=slices_FP[i]+[slices_FP[i][-1]+1] #merge two lists into new one
#                     for i in range(slices_TP.count(slices_FP[i][-1]+1)):                                            
#                         # slices_TP_new.append(new_nod)
                    
#                         slices_TP_new.append(new_nod)
#                         # print("e newnod {}".format(new_nod))
                        
#                         if slices_FP[i][-1]+1 in slices_FN:
#                             while True:
#                                 slices_FN.remove(slices_FP[i][-1]+1)
           
            
#             elif slices_FP[i][0]-1 in slice_vals: 
#                     print("INTO SECOND ELIF")
#                     new_nod=[slices_FP[i][0]-1]+slices_FP[i] #merge two lists into new one
#                     if slices_TP.count(slices_FP[i][0]-1)!=0: #!!if-else since true nodule not kept in FP and not added in TP in 383275
#                         for i in range(slices_TP.count(slices_FP[i][0]-1)):                                            
#                             # slices_TP_new.append(new_nod)
                        
#                             slices_TP_new.append(new_nod)
#                             # print("f newnod {}".format(new_nod))
                            
#                             if slices_FP[i][0]-1 in slices_FN:
#                                 while True:
#                                     slices_FN.remove(slices_FP[i][0]-1)
#                     else:
#                         slices_TP_new.append(new_nod)
#                         if slices_FP[i][0]-1 in slices_FN:
#                                 while True:
#                                     slices_FN.remove(slices_FP[i][0]-1)
                
#                 # print(slices_FP[i])
#             else:
#                 # print('not {}'.format(slices_FP[i]))
#                 # print('final else loop')
#                 slices_FP_new.append(slices_FP[i]) #else it is indeed FP
                
#         except:
#             print('ERROR: No FP slices or other that should be checked!')
#             import traceback
#             traceback.print_exc()
#             pass #!!! check if some condition shoudl be dfinedhere
            
        
        
#         # if slices_FP[i][-1]+1 in slices_TP or slices_FP[i][0]-1 in slices_TP:
#         #     print('normal')
#         #     print(slices_FP)
#         #     try:
#         #         if slices_FP[i][-1]+1 in slices_TP and slices_FP[i+1][0]-1 in slices_TP:
#         #             new_nod=slices_FP[i]+[slices_FP[i][-1]+1]+slices_FP[i+1]
#         #             slices_TP_new.append(new_nod)
#         #     except:
#         #         pass
            
#         #     # print(slices_FP[i])
#         # else:
#         #     print('not {}'.format(slices_FP[i]))
#         #     slices_FP_new.append(slices_FP[i])
            
#     # print('FPs before new {}'.format(slices_FP))
#     # print('TPs before new {}'.format(slices_TP))    

#         #!!!!!
#     # print('FPs after new {}'.format(slices_FP_new))  
#     # print('TPs after new {}'.format(slices_TP_new))                                 
#     slices_FP=slices_FP_new    
#     slices_TP=slices_TP_new
#     # slices_TP=combine_slices(slices_TP)
    

#     # try:
#     #     if isinstance(slices_FP[0],list):
#     #         pass
#     #     else:
#     #         slices_FP=[slices_FP]
#     # except:
#     #     pass
        
#     # # try:
#     # #     slices_TP==[]
#     # # except:
#     # #     if isinstance(slices_TP[0],list):
#     # #         pass
#     # #     else:
#     # #         slices_TP=[slices_TP]
#     # try:
#     #     if isinstance(slices_TP[0],list):
#     #         pass
#     #     else:
#     #         slices_TP=[slices_TP]
#     # except:
#     #     pass
    
    
#     print('Final list of TP slices is {}'.format(slices_TP))
    
#     # print(confirmed_tp_fp)
#     #FAILURE TO FIND FP IN 892519 AND IN OTHERS 'ERROR NOT FIXED'        
#     for in_num,lis in enumerate(slices_TP): 
#         for elem in lis:
#             if elem in confirmed_tp_fp:
#                 slices_FP.append(lis)
#                 print('Fp append for error similar to 892519')
                
#                 #!!!!!!!!!error in 670208
#     # for ind,miss in enumerate(possible_missed_FP_slices): #slices with >1 nods
#     #     # tot_FP=np.sum([x.count(miss) for x in slices_FP]) 
#     #     tot_TP=np.sum([x.count(miss) for x in slices_TP])
#     #     tot_slices=tot_TP #tot_FP+ #Total num of slices that contain this slice
#     #     find_ind=[miss in TP for TP in slices_TP] #indices in TP where this slice exist
#     #     for additional in list(np.asarray(slices_TP)[np.where(np.asarray(find_ind)>0)]):
#     #         # while FP_nods[ind]>tot_slices:
#     #             slices_FP.append(additional)
#     #             tot_slices=tot_slices+1
#     #             if FP_nods[ind]==tot_slices:
#     #                 break
#     #             else:
#     #                 continue
                        
#     check_FP=[]
#     for key,val in AI_num_nods[index_init].items():
#         num_occur_tp=np.sum([x.count(int(key.split('.')[4])) for x in slices_TP])
#         num_occur_fp=np.sum([x.count(int(key.split('.')[4])) for x in slices_FP])
#         if val>1:
            
#             # if val>num_occur_tp:
#                 while val>num_occur_tp:
#                     check_FP.append(size_AI[index_init]-int(key.split('.')[4]))
#                     # print("checking {} for FP in tp loop".format(size_AI[index_init]-int(key.split('.')[4])))
#                     num_occur_tp=num_occur_tp+1
                
#             # if val>num_occur_fp:
#                 while val>num_occur_fp:
#                     check_FP.append(size_AI[index_init]-int(key.split('.')[4]))
#                     # print("checking {} for FP in fp loop".format(size_AI[index_init]-int(key.split('.')[4])))
#                     num_occur_fp=num_occur_fp+1        

#     print("CHECKED FP ARE {}".format(check_FP))
#     check_FP=np.unique(check_FP)
#     check_FP.sort()
#     check_FP=combine_slices(check_FP)
#     print("CHECKED FP after combi ARE {}".format(check_FP))
    
#     # for i in range(len(check_FP)):
#     #     check_FP.count
#     # if hh
# #!!!BUT IF ACTIVATED PROBLEM WITH       507704, 633549 IN MOD!!          
#     # if len(slices_FP)!=1: #SINCE 998310 CREATES LISTS OF MANY FP WHILE ITS ONLY 1!
#     #     slices_FP=list(np.unique(slices_FP))  
                
#     # print('Number of nodules in TP list above is {}'.format(len(slices_TP)))
#     #!!!!CHECK IF MORE THAN ONE NODULES IN THESE SLICES!! SO 2 FP THEN!!
#     print('Final list of FP slices is {}'.format(slices_FP))
#     # print('Number of nodules in FP list above is {}'.format(len(slices_FP)))

#     # print('Final list of FN slices is {} and the number of nods=length is {}'.format(slices_FN,len(slices_FN)))
#     FNs_last=slices_FN+possible_FNs
#     #possible_FNs non-zero only when all elements not found in slice_vals are not considered as FPs
#     # print("slices FN are {}".format(slices_FN))
#     # print("possible Fns are {}".format(possible_FNs))
#     FN_change=[]
#     for i in FNs_last:
#         if isinstance(i,list) and len(i)==1: #if it's a list with only one element/slice
#             for j in i:
#                 FN_change.append(j)
#         elif isinstance(i,list) and len(i)>1:
#             FN_change.append(i)
#         else:
#             FN_change.append(i) #if not a list then add it to our list
    
#     # print("all FNs before unique list are {}".format(FN_change))
#     FN_change_test=FN_change #just to print below - DEL
    
#     try:
#         if isinstance(FN_change[0],list)!=True:
#             FN_change=list(np.unique(FN_change))
#     except:
#         pass
    
#     # if len(FN_change_test)!=len(FN_change): #WHEN NO AI OUTPUTS BUT VALS HAS VALUES - GOT IT FOR 101191
#     #         print("all FNs before unique list are {}".format(FN_change_test))

#     # missed_FNs=[]
#     # for i in slice_vals:
#     #     try:
#     #         if isinstance[slices_TP[0],list]:
#     #             for list_tp in slices_TP:
#     #                 if i in list_TP:
#     #                     continue
#     #                 else:
#     #                     try:
#     #                         if isinstance[slices_FP[0],list]:
#     #                             for list_fp in slices_FP:
#     #                                 if i in list_fp:
#     #                                     continue
#     #                                 else:
#     #                                     missed_FNs.append(i)
                                        
    
#     print('Final list of FN slices is {}'.format(FN_change))#slices_FN+possible_FNs))
#     # print('Number of nodules in FN list above is {}'.format(len(FN_change)))#len(slices_FN)+len(np.unique(possible_FNs))))


#         # print('We have {} TP, {} FP, and {} FN'.format(len((slices_TP)),len(np.unique(slices_FP)),len(np.unique(slices_FN))+len(np.unique(possible_FNs))-num_remove))
#         #TP,FP,FN))  #+len(list(set(slice_vals)-set(slice_vals_found)))
#         #!!!np.unique in TP slices below!!!!!!!
#         # print("Total num of TP,FP is {}".format(len((slices_TP))+len(np.unique(slices_FP))))
#         # print('Total AI nodules found: {}'.format(len(np.unique([num.split('.')[4] for num in list(AI_nodules[index_init].keys())]))))
#     # print('TP Slices may also contain additional FP or even more than 1 TP! Need OCR to confirm!')
#     try:
#         assert len(slices_TP)<=len(slice_vals)
#     except:
#         print("ERROR! TP list has {} nods and should be less or equal to the {} of ground truth/slice_vals".format(len(slices_TP),len(slice_vals)))
#     # print('TP list has {} nods and should be less or equal to the {} of ground truth/slice_vals'.format(len(slices_TP),len(slice_vals)))
#     print('True nodules from REDCap in slices: {}'.format(slice_vals))
    
#     print('Number of nodules in FP list above is {}'.format(len(slices_FP)))
    
#     try:
#         assert len(slices_TP)+len(FN_change)==len(slice_vals)
#     except:
#         print('ERROR! Number of TP and FN not the same as number of manual annotations in REDCap')
#         print('There are {} TP, {} FN, and REDCap has {} annotations'.format(len(slices_TP),len(FN_change),len(slice_vals)))
        
#     # print('Number of nodules in FN list above is {}'.format(len(FN_change)))#len(slices_FN)+len(np.unique(possible_FNs))))
    
#     fn_slices=[]
#     for i in FN_change:
#         for index,val in enumerate(slice_vals):
#             if i==val:
#                 FN_volumes.append(volume_ids[index])
#                 fn_slices.append(val)
    
#     tp_slices=list(slice_vals)
#     for el in fn_slices:
#         try:
#             tp_slices.remove(el)
#         except:
#             pass
        
              
#     ind_check=[]#to avoid problems like 670208
#     for i in tp_slices:
#         for index2,val2 in enumerate(slice_vals):
#             if i==val2 and index2 not in ind_check:
#                 TP_volumes.append(volume_ids[index2])
#                 ind_check.append(index2)
#                 break
                
    
#     print("Volumes of TP nodules found are {}".format(TP_volumes))
#     print("Volumes of FN nodules not found are {}".format(FN_volumes))
    
#     pat_vols={'tp':TP_volumes,'fn':FN_volumes}#,'patient':path.split('/')[-1]}
#     all_volumes.append(pat_vols)
#     FP_num.append(len(slices_FP))

#     if len(slices_TP)+len(FN_change)<len(slice_vals):
#         print("Nodule(s) missed probably because the same slice contains 2 new nodules in manual annotations")
#     # print('Num of Total nodules expected from ground truth is {}'.format(len(slice_vals)))    
       
#     #!!!!                
#     if len(confirmed_tp_fp)!=0:
#         print('ERRORS IN THAT NOT FIXED!!!!!!')
#   #!!! if num_nods>1??             

#     print('\n')
#     print('TP Slices may also contain additional FP or even more than 1 TP! Need OCR to confirm!')  
#     print("The above volumes with the same order as nodule slices in REDCap")
       

#     end=time.time()
#     print("Time it took to run full code is {} secs".format(end-start))
    
#     sys.stdout.close()
#     sys.stdout = sys.__stdout__  


# sys.stdout = open(output_path+'final_results.txt', 'w') #Save output here
# print("Patient names are {}".format(patient_names))
# print("Volumes of the above patients for TP and FN are {}".format(all_volumes))
# print("Number of FP for each of the above patients is {}".format(FP_num))

# print("Total number of patients is {}".format(len(patient_names)))
# print("Total number of volume information is {}".format(len(all_volumes)))
# print("Total number of FP patients is {}".format(len(FP_num)))


# sys.stdout.close()
# sys.stdout = sys.__stdout__  
#!!!! FAILS WHEN MORE THAN 10 NODS, IF MANY NODS WITH SAME Y OR X COORD, IF FP AND TP ON SAME SLICE  
#!!!! NODULES SLICES NOT EXACT AS IN AI BUT ALWAYS INCLUDES SLICES WITHIN THE RANGE OF DETECTIONS OF AI!! NOT ALWAYS!! 670208!!  
#!!!! ONLY WAY TO RELATE L01 ETC OF AI TO L01 OF MANUAL ANNOTS IS WITH OCR OR USING LOG FILES - GERTJAN !!!!

# end=time.time()
# print("Time it took to run full code is {} secs".format(end-start))
        
# sys.stdout.close()
# sys.stdout = sys.__stdout__            