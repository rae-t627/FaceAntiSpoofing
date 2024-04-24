We have slightly modified the implementation given in [SSDG-CVPR2020](https://github.com/taylover-pei/SSDG-CVPR2020) to suit our needs. 

## data_label/generate_label.py

- Modified the functions to load a particular dataset, to fit our datasets accordingly. 
- Modified the replay_attack_process() to load the dataset by extracting the frames from a given video.
- Created a function crop_face() to perform Haar Cascade Face Detection Algorithm on the images to focus only on the faces.
- Created separate functions to load the different datasets both for cropped and uncropped dataset.

## experiments/L_N_to_R or L_R_to_N or R_N_to_L/train_ssfg_full.py 

- Modified the whole code to work for three datasets instead of the original four datasets.

## experiments/L_N_to_R or L_R_to_N or R_N_to_L/dg_test.py 

- Altered the way the accuracy is being calculated for the model.
- Made a script to create the ROC Curve after testing the model on a dataset.

## utils/dataset.py
- included histogram equalization and LBP pre-processing

## utils/get_loader.py

- changed the function to work with 2 datasets for training and the third for testing.
- Also wrote our own get_points function

## utils/utils.py

- Created several functions to generate the video ids for each dataset.
- Modified the sample_frames function to be called get_points, so that the images from the same person or video be given the same label, which would be used for training the model. 
- Modified the accuracy function
- Modified the save_checkpoint function to also save the optimizer state at a particular point.

## tsne/tsne.py
- loaded the data
- Generated the features
- created the tsne plots

## loss/AdLoss.py
- tailored the loss to 2 datasets