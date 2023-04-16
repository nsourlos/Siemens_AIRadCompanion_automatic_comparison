# Siemens AIRadCompanion - semi automated comparison with Radiologists’ annotations

![Alt text](./siemens-ai-rad-companion.svg)

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/uses-badges.svg)](https://forthebadge.com)

[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-no-red.svg)]( https://github.com/nsourlos/Siemens_AIRadCompanion_automatic_comparison)


> This tool can be used to compare the Siemens AI Rad Companion nodule detection software output with the one of the radiologists (ground truth). The python file that does that can be found [here](/automated_nodule_comparison.py)

For this tool to run, the following files should be used first:

- A file that exports from REDCap the actual findings annotated by the radiologists. This file creates one csv file per participant and can be found [here](/redcap_patient_extraction.py) 

######...........

####EXPLAIN HOW TO DONWLOAD SCANS AND THE FORMAT (FOLDERS/SUBFOLDERS) THAT SHOULD BE USED!!



#### Files to create xlsx file with AI nodule information

To create the xlsx file used in the automation process described above we used this [file](/AI_aorta_calcium_lungnodules_emphysema_vertebra_fat_to_REDCap.ipynb). This extracts information from the output txt files of the AI. An example of such a txt file with only nodule information can be found [here](1.2.276.0.28.3.345049594267.42.7220.20220705105655000.txt). We might have different txt files for the same participant, each one sent to AI for different reasons (nodule detection, aorta measurements etc.). At last, this file also uses some dictionaries (in the form of pickle objects) that store information of the files that correspond to each participants. This is because we might have more than one files for each participant, each sent to AI for a different reason (nodule detections, aorta measurements etc.) and some of them may also contain errors (eg. missing slices). The pickle dictionaries were created using this [file](/AI_timestamp.ipynb) which uses some other txt files with information about the time it took for the AI to process a scan (nodule detection, aorta measurements etc.) as well as the number of slices processed (which can be used to identify possible errors, if not all of them were sent). The pickle dictionaries returned, have each participant id as well as the name of the most recent file or the file with the most slices that we want to extract. The name of the file with information about the duration of each task will be the same as the name of the file from which information about this task exists and from which it will be extracted from. An example of a file with the duration of each task can be found [here](1.2.276.0.28.3.345049594267.42.2584.20221028035850000.txt). Structure is similar for other files as well. 


# [automated_nodule_comparison.py](/automated_nodule_comparison.py)

The following paths need to be specified:

- Path of folder containing subfolders with anonymized participants ids (6 digit numbers). These subfolders should contain DICOM files of at least the following:
 - The original CT scan of the participant (many 2D slices, one file for each slice)

An example of a CT DICOM slice can be seen in the screenshot below:

<img src="./images/CT_scan_slice.png">

 - The CT scan with the AI annotations/detections overlayed on top of it

An example of a AI DICOM slice with a nodule can be seen in the screenshot below:

<img src="./images/AI_scan_slice.png">

 - Some 2D slices with the nodule contour (Radiologists’ annotations on top of the original CT scan slice – will be referred as annotated CT files from now on)

An example of a DICOM slice with the nodule contour outlined (changed in a proper colormap for visualization) can be seen in the screenshot below:

<img src="./images/results_annotation.png">

 - If exist, some files with 3D segmentation masks of the nodules 
 - If exist, any other non-CT data (eg. structural reports). These will not be used by our tool/processing pipeline

- Path of folder in which the output images and txt files will be saved
- Path of folder that contains one csv file of each participant with the slices that actually contain nodules. These were taken from REDCap (where radiologists stored nodule information). These files should contain the following columns:
  - A column with `slice` in its name with the slice of the original scan in which the nodule can be found
  - A column with `nodule_id` in its name containing the id of the nodule for each slice
  - Two columns with `volume_solid` and/or `volume_subsolid` in their name, containing information about the nodule component (if it has solid/subsolid parts)


- Path of an xlsx file with slices in which AI detected nodules. This file was created using the AI generated txt files that contain that information. This file contains one column with the participant id (first column), 10 columns with the volumes of each nodule found on that participant (the next 10 columns) – volumes refer to both the solid and subsolid components combined -, 10 columns with the positions of the relative position of the participant in the table in which the nodules were found (next 10 columns) – will be converted to slice numbers in the script -, and a final column with the total number of nodules for each participant. If a participant has less than 10 nodules then some fields will be filled with `-`.

**The jupyter notebook that was used to create the above xlsx file is also provided and can be found [here](/AI_aorta_calcium_lungnodules_emphysema_vertebra_fat_to_REDCap.ipynb). This file also uses [this](/AI_timestamp.ipynb)** to get some files needed to run.

## Brief description of how this tool works

The algorithm will check the folder with the DICOM files and will determine which of those correspond to the original CT scan, which to the AI output scan (overlayed with nodule contours), which to the annotated CT scan slices with the nodule contours, and, if exist, which files contain a SEG mask. 

It will then keep in separate variables which CT slices of the original scan have annotated slices with nodules. It also will keep track of which AI slices might contain nodules.

Since the above annotated slices with annotations of the radiologists might also contain slices which were not nodules, a further check should be performed to confirm if a finding is a nodule or not. Similarly, because we might have more than one nodule in each AI scan slice, a further check should be performed to distinguish where each nodule can be found, along with its id. That information is provided in the csv files of the participants and in the xlsx file for the radiologists and for the AI annotations respectively.

The problem is that an annotated CT slice may not be on the same slice as the GT slice in the csv file. The reason for that is that a nodule might be extended in more than one slices and radiologists outlined it in one slice and might have added it in a different slice in REDCap (from which the csv was created). This is why we also check some of the nearby slices and compare them with our annotated slice. We similarly have to check some of the nearby slices from which AI detected nodules to make sure that we don’t confuse TP and FP.

After finding which of the annotated CT slices actually contain nodules, and in which slice these nodules actually are, the main idea is to compare their contour with the contour of the same AI slice, if exists. If there is overlap, then this nodule would be a TP. If not, then a FN. In cases of AI detected nodules for which we don’t have any correspondance with an annotated CT slice, this nodule would be considered as FP. 

More specific details, to address problems (eg. 1 TP and 1 FN nodule in the same slice) can be seen in the code. The main difficulty faced was to correlate each detection with the respective nodule id.

Images of the FPs and FNs are also saved to be used for a blinded review by some extra radiologists (not being aware if it’s a FP –and therefore an AI detection- or a FN –and therefore a detection of other radiologists-).

#### Failures of the algorithm

This algorithm might fail to correctly identify at least one nodule in the following cases (for which the output is just an empty line):

1. When a participant has more than 10 nodules. This would happen because AI will detect the extra nodules and place a red rectangle bounding box around them. It is not possible with the procedure described in this code to identify which are the bounding boxes that correspond to the 10 nodules that were reported by AI and which those of extra detections. These extra detections are nodules in which a lower confidence score was given to them by the AI.
2. Sometimes the AI slice from the xlsx file does not correspond to a slice that actually contains a nodule (close to it but not a slice with a red nodule contour). The algorithm will fail for those cases.
3. If the same AI slice contains more than one nodule, and one of them is TP and the other FP, the resulting images may not be found correctly.
4. In cases in which the GT slice is different from the one in the annotated CT slices we might have issues.
5. At last, the most important failure is the one that **will not raise an error**. This might happen when there are errors in the xlsx file id numbers. If for example, radiologists accidentaly noticed that the nodule with id 1 is in slice 277 while the nodule with id 2 is in slice 329 while it’s vice versa, our algorithm will not raise any errors. These cannot be avoided and this is why a second manual verification of findings is needed. This is the same as confirming that the information in the database is correct. On some other cases errors might be raised (empty fields in the output excel file). These cases should be checked and filled manually.


###############
Below some failure cases extracted from comments in the code:

Failure cases when same slice more than two times in GT slices
TP images are not extracted properly for now
Might fail if GT slice in original_CTs (not in 'Results') and not any AI nodules
If two FN in the same slice it might fail
Probably fails when one TP and one FP in the same slice (as in 673634) 
Fails when 3 consecutive FP slices

## Detailed description of functions used by this algorithm

After setting the above mentioned paths, the first steps of our algorithm are the following:

- It saves in different variables all CT files (AI, original scan files, annotated CT image names), as well as the total slices in the SEG files, in AI output, and the total num of CT slices (original scan). It also saves in a txt file information of possible errors in them. It is assumed that the input path contains only DICOM files of a specific participant.
- Correlates annotated CT slices with slices of the original scan. This means that it finds the slice number of the annotated 2D slice with the nodule contour. It also finds if there are any empty CT slices and files with possible missed nodules due to low threshold (set to distinguish which CT slice is similar to one in annotated CT slices).
- Plots the nodules in the SEG files and saves in new variables files with the right order. These are SEG files and their corresponding original CT slices, and annotated CT slices. It also provides information of the SEG slices with errors.

After having the above, we end up with a few variables of interest that will be used to get our final results:
- CT scan slices with nodules
- Annotated slices of the above CT slices with the nodule contour
- Possible segmentation masks that correspond to the above nodules
- Possible errors in the above segmentation masks

Now, our algorithm will extract information from the csv files of each participant to find which slices might contain nodules, as well as their nodule ids and their volume. These are the actual nodules slices. The annotated CT slices we have until know might also contain findings which are not nodules and were just noted by the radiologists.

Similarly, it will find in which slices there are AI detected nodules and their slice number, id, and their volume, as quantified by the AI. This will be done using the xlsx file that was created with the output txt files of the AI algorithm. This file is created with this [file](/AI_aorta_calcium_lungnodules_emphysema_vertebra_fat_to_REDCap.ipynb) which also needs this [file](/AI_timestamp.ipynb) to work.

At last, some screenshots of when a nodule is a TP, a FP and a FN are presented below:

![Alt text](/images/L6_298_TP_585377_0-100.jpg?raw=true "A TP nodule")
![Alt text](/images/L7_248_FN_585377_0-100.jpg?raw=true "A FN nodule")
![Alt text](/images/L8_AI_286_FP_585377_0-100.jpg?raw=true "A FP nodule")




## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

 
## License
[MIT License](LICENSE)
