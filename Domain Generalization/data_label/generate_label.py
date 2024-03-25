import os
import json
import sys
import glob
import cv2
import numpy as np
from PIL import Image
# change your data path
data_dir = 'Dataset/'

def lcc_fasd_process():
    train_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './lcc_fasd/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    
    # Adjust the paths and file names accordingly
    dataset_path = data_dir + 'LCC_FASD/'
    train_real_path = dataset_path + 'LCC_FASD_training/real/*.png'
    # print(train_real_path)
    train_fake_path = dataset_path + 'LCC_FASD_training/spoof/*.png'
    test_real_path = dataset_path + 'LCC_FASD_evaluation/real/*.png'
    test_fake_path = dataset_path + 'LCC_FASD_evaluation/spoof/*.png'
    
    # Load train data
    train_real_files = glob.glob(train_real_path)
    # print(train_real_files)
    train_fake_files = glob.glob(train_fake_path)
    for file_path in train_real_files:
        train_final_json.append({'photo_path': file_path, 'photo_label': 1})
        all_final_json.append({'photo_path': file_path, 'photo_label': 1})
        real_final_json.append({'photo_path': file_path, 'photo_label': 1})
    for file_path in train_fake_files:
        train_final_json.append({'photo_path': file_path, 'photo_label': 0})
        all_final_json.append({'photo_path': file_path, 'photo_label': 0})
        fake_final_json.append({'photo_path': file_path, 'photo_label': 0})
        
    # Load test data
    test_real_files = glob.glob(test_real_path)
    test_fake_files = glob.glob(test_fake_path)
    for file_path in test_real_files:
        test_final_json.append({'photo_path': file_path, 'photo_label': 1})
        all_final_json.append({'photo_path': file_path, 'photo_label': 1})
        real_final_json.append({'photo_path': file_path, 'photo_label': 1})
    for file_path in test_fake_files:
        test_final_json.append({'photo_path': file_path, 'photo_label': 0})
        all_final_json.append({'photo_path': file_path, 'photo_label': 0})
        fake_final_json.append({'photo_path': file_path, 'photo_label': 0})
    
    # Print statistics
    print('\nLCC_FASD: ', len(all_final_json))
    print('LCC_FASD(train): ', len(train_final_json))
    print('LCC_FASD(test): ', len(test_final_json))
    print('LCC_FASD(all): ', len(all_final_json))
    print('LCC_FASD(real): ', len(real_final_json))
    print('LCC_FASD(fake): ', len(fake_final_json))
    
    # Write to JSON files
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()

def nuaa_process():
    train_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './nuua/'
    
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
        
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    
    # Read train and test face lists from text files
    dataset_path = data_dir + 'NUAA/Detectedface/'
    # client_train_file = os.path.join(dataset_path, 'client_train_face.txt')
    client_train_file = '/home/eeiith/Desktop/Project1/Kaustubh/IVP/SSDG-CVPR2020/data_label/Dataset/NUAA/raw/client_train_raw.txt'
    client_test_file = '/home/eeiith/Desktop/Project1/Kaustubh/IVP/SSDG-CVPR2020/data_label/Dataset/NUAA/raw/client_test_raw.txt'
    imposter_train_file = '/home/eeiith/Desktop/Project1/Kaustubh/IVP/SSDG-CVPR2020/data_label/Dataset/NUAA/raw/imposter_train_raw.txt'
    imposter_test_file = '/home/eeiith/Desktop/Project1/Kaustubh/IVP/SSDG-CVPR2020/data_label/Dataset/NUAA/raw/imposter_test_raw.txt'
    # client_test_file = os.path.join(dataset_path, 'client_test_face.txt')
    # imposter_train_file = os.path.join(dataset_path, 'imposter_train_face.txt')
    # imposter_test_file = os.path.join(dataset_path, 'imposter_test_face.txt')
    
    # Load train data
    with open(client_train_file, 'r') as f:
        client_train_faces = f.read().splitlines()
    with open(imposter_train_file, 'r') as f:
        imposter_train_faces = f.read().splitlines()
    
    client_train_faces = [i.replace("\\", "/") for i in client_train_faces]
    imposter_train_faces = [i.replace("\\", "/") for i in imposter_train_faces]
        
    for face in client_train_faces:
        file_path = os.path.join(dataset_path, 'ClientFace', face)
        
        train_final_json.append({'photo_path': file_path, 'photo_label': 1})
        all_final_json.append({'photo_path': file_path, 'photo_label': 1})
        real_final_json.append({'photo_path': file_path, 'photo_label': 1})
        
    for face in imposter_train_faces:
        file_path = os.path.join(dataset_path, 'ImposterFace', face)
        
        train_final_json.append({'photo_path': file_path, 'photo_label': 0})
        all_final_json.append({'photo_path': file_path, 'photo_label': 0})
        fake_final_json.append({'photo_path': file_path, 'photo_label': 0})
        
    # Load test data
    with open(client_test_file, 'r') as f:
        client_test_faces = f.read().splitlines()
    with open(imposter_test_file, 'r') as f:
        imposter_test_faces = f.read().splitlines()
        
    client_test_faces = [i.replace("\\", "/") for i in client_test_faces]
    imposter_test_faces = [i.replace("\\", "/") for i in imposter_test_faces]
      
    for face in client_test_faces:
        file_path = os.path.join(dataset_path, 'ClientFace', face)
        test_final_json.append({'photo_path': file_path, 'photo_label': 1})
        all_final_json.append({'photo_path': file_path, 'photo_label': 1})
        real_final_json.append({'photo_path': file_path, 'photo_label': 1})
    for face in imposter_test_faces:
        file_path = os.path.join(dataset_path, 'ImposterFace', face)
        test_final_json.append({'photo_path': file_path, 'photo_label': 0})
        all_final_json.append({'photo_path': file_path, 'photo_label': 0})
        fake_final_json.append({'photo_path': file_path, 'photo_label': 0})
    
    # Print statistics
    print('\nNUAA Detectedface: ', len(all_final_json))
    print('NUAA Detectedface(train): ', len(train_final_json))
    print('NUAA Detectedface(test): ', len(test_final_json))
    print('NUAA Detectedface(all): ', len(all_final_json))
    print('NUAA Detectedface(real): ', len(real_final_json))
    print('NUAA Detectedface(fake): ', len(fake_final_json))
    
    # Write to JSON files
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()

def replay_attack_process():
    train_final_json = []
    valid_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './replay_attack/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    
    dataset_path = data_dir + 'Replay Attack/Dataset/'
    # Traverse through the dataset directory
    for subset in ['train', 'test', 'devel']:
        subset_dir = os.path.join(dataset_path, subset)
        for attack_type in ['real', 'attack']:
            attack_dir = os.path.join(subset_dir, attack_type)
            for video_folder in os.listdir(attack_dir):
                video_path = os.path.join(attack_dir, video_folder)
                # Open the video file
                cap = cv2.VideoCapture(video_path)
                # Capture random frames until the end of the video
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Process frame here, for example, save it to a directory
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) % 25 == 0:  
                        frame_path = os.path.join('frames',video_path[:-4], f'frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg')
                        # print(frame_path)
                        if not os.path.exists(os.path.join('frames',video_path[:-4])):
                            os.makedirs(os.path.join('frames',video_path[:-4]))
                        cv2.imwrite(frame_path, frame)
                        # Determine label based on attack_type
                        if attack_type == 'real':
                            label = 1
                        else:
                            label = 0
                        # Append frame path and label to respective lists
                        dict = {'photo_path': frame_path, 'photo_label': label}
                        all_final_json.append(dict)
                        if subset == 'train':
                            train_final_json.append(dict)
                        elif subset == 'devel':
                            valid_final_json.append(dict)
                        else:
                            test_final_json.append(dict)
                        if label == 1:
                            real_final_json.append(dict)
                        else:
                            fake_final_json.append(dict)
                cap.release()
                
            # Check for fixed and hand directories
            fixed_dir = os.path.join(attack_dir, 'fixed')
            hand_dir = os.path.join(attack_dir, 'hand')
            
            if os.path.exists(fixed_dir):
                for video_folder in os.listdir(fixed_dir):
                    video_path = os.path.join(fixed_dir, video_folder)
                    # Open the video file
                    cap = cv2.VideoCapture(video_path)
                    # Capture random frames until the end of the video
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Process frame here, for example, save it to a directory
                        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 25 == 0:  
                            frame_path = os.path.join('frames',video_path[:-4], f'frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg')
                            # print(frame_path)
                            if not os.path.exists(os.path.join('frames',video_path[:-4])):
                                os.makedirs(os.path.join('frames',video_path[:-4]))
                            cv2.imwrite(frame_path, frame)
                            # Determine label based on attack_type
                            if attack_type == 'real':
                                label = 1
                            else:
                                label = 0
                            # Append frame path and label to respective lists
                            dict = {'photo_path': frame_path, 'photo_label': label}
                            all_final_json.append(dict)
                            if subset == 'train':
                                train_final_json.append(dict)
                            elif subset == 'devel':
                                valid_final_json.append(dict)
                            else:
                                test_final_json.append(dict)
                            if label == 1:
                                real_final_json.append(dict)
                            else:
                                fake_final_json.append(dict)
                    cap.release()
            
            if os.path.exists(hand_dir):
                for video_folder in os.listdir(hand_dir):
                    video_path = os.path.join(hand_dir, video_folder)
                    # Open the video file
                    cap = cv2.VideoCapture(video_path)
                    # Capture random frames until the end of the video
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Process frame here, for example, save it to a directory
                        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 25 == 0:  
                            frame_path = os.path.join('frames',video_path[:-4], f'frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg')
                            # print(frame_path)
                            if not os.path.exists(os.path.join('frames',video_path[:-4])):
                                os.makedirs(os.path.join('frames',video_path[:-4]))
                            cv2.imwrite(frame_path, frame)
                            # Determine label based on attack_type
                            if attack_type == 'real':
                                label = 1
                            else:
                                label = 0
                            # Append frame path and label to respective lists
                            dict = {'photo_path': frame_path, 'photo_label': label}
                            all_final_json.append(dict)
                            if subset == 'train':
                                train_final_json.append(dict)
                            elif subset == 'devel':
                                valid_final_json.append(dict)
                            else:
                                test_final_json.append(dict)
                            if label == 1:
                                real_final_json.append(dict)
                            else:
                                fake_final_json.append(dict)
                    cap.release()
    # Print statistics
    print('\nReplay Attack: ', len(all_final_json))
    print('Replay Attack(train): ', len(train_final_json))
    print('Replay Attack(valid): ', len(valid_final_json))
    print('Replay Attack(test): ', len(test_final_json))
    print('Replay Attack(all): ', len(all_final_json))
    print('Replay Attack(real): ', len(real_final_json))
    print('Replay Attack(fake): ', len(fake_final_json))
    
    # Write to JSON files
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()
    
def crop_face(image):
    # Convert Pillow image to numpy array
    img_array = np.array(image)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Load face cascade classifier
    face_classifier = cv2.CascadeClassifier('/home/eeiith/Desktop/Project1/Kaustubh/IVP/SSDG-CVPR2020/utils/haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # If no faces detected, return original image
    if len(faces) == 0:
        return image
    
    # Crop and return the first detected face
    x, y, w, h = faces[0]
    cropped_face = image.crop((x - 10, y - 10, x+w + 10, y+h + 10))
    
    return cropped_face

def save_cropped_images():
    # Load the JSON files
    with open('./lcc_fasd/all_label.json', 'r') as f:
        lcc_fasd_data = json.load(f)
    with open('./nuua/all_label.json', 'r') as f:
        nuua_data = json.load(f)
    with open('./replay_attack/all_label.json', 'r') as f:
        replay_attack_data = json.load(f)
    
    # Save cropped images for LCC_FASD
    for i in range(len(lcc_fasd_data)):
        image = Image.open(lcc_fasd_data[i]['photo_path'])
        photo_name = lcc_fasd_data[i]['photo_path'].split('/')[-1]
        photo_path_list = lcc_fasd_data[i]['photo_path'].split('/')[1:]
        # Create a folder path by joining all elements of the list except the last one
        photo_folder = '/'.join(photo_path_list[:-1]) + '/'
        # Join all the elements of a list using 
        photo_folder = './lcc_fasd_cropped/' + photo_folder
        photo_path = photo_folder + photo_name
        # print(photo_path)
        cropped_image = crop_face(image)
        if not os.path.exists(photo_folder):
            os.makedirs(photo_folder)
        cropped_image.save(photo_path)

    # Save cropped images for NUAA
    for i in range(len(nuua_data)):
        image = Image.open(nuua_data[i]['photo_path'])
        photo_name = nuua_data[i]['photo_path'].split('/')[-1]
        photo_path_list = nuua_data[i]['photo_path'].split('/')[1:]
        # Create a folder path by joining all elements of the list except the last one
        photo_folder = '/'.join(photo_path_list[:-1]) + '/'
        # Join all the elements of a list using 
        photo_folder = './nuua_cropped/' + photo_folder
        photo_path = photo_folder + photo_name
        # print(photo_path)
        cropped_image = crop_face(image)
        if not os.path.exists(photo_folder):
            os.makedirs(photo_folder)
        cropped_image.save(photo_path)
    
    # Save cropped images for Replay Attack
    for i in range(len(replay_attack_data)):
        image = Image.open(replay_attack_data[i]['photo_path'])
        photo_name = replay_attack_data[i]['photo_path'].split('/')[-1]
        photo_path_list = replay_attack_data[i]['photo_path'].split('/')[1:]
        # Create a folder path by joining all elements of the list except the last one
        photo_folder = '/'.join(photo_path_list[:-1]) + '/'
        # Join all the elements of a list using 
        photo_folder = './replay_attack_cropped/' + photo_folder
        photo_path = photo_folder + photo_name
        # print(photo_path)
        cropped_image = crop_face(image)
        if not os.path.exists(photo_folder):
            os.makedirs(photo_folder)
        cropped_image.save(photo_path)
        
def lcc_fasd_process_cropped():
    train_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './lcc_fasd/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label_crop.json', 'w')
    f_test = open(label_save_dir + 'test_label_crop.json', 'w')
    f_all = open(label_save_dir + 'all_label_crop.json', 'w')
    f_real = open(label_save_dir + 'real_label_crop.json', 'w')
    f_fake = open(label_save_dir + 'fake_label_crop.json', 'w')
    
    # Adjust the paths and file names accordingly
    dataset_path = 'lcc_fasd_cropped/LCC_FASD/'
    train_real_path = dataset_path + 'LCC_FASD_training/real/*.png'
    # print(train_real_path)
    train_fake_path = dataset_path + 'LCC_FASD_training/spoof/*.png'
    test_real_path = dataset_path + 'LCC_FASD_evaluation/real/*.png'
    test_fake_path = dataset_path + 'LCC_FASD_evaluation/spoof/*.png'
    
    # Load train data
    train_real_files = glob.glob(train_real_path)
    # print(train_real_files)
    train_fake_files = glob.glob(train_fake_path)
    for file_path in train_real_files:
        train_final_json.append({'photo_path': file_path, 'photo_label': 1})
        all_final_json.append({'photo_path': file_path, 'photo_label': 1})
        real_final_json.append({'photo_path': file_path, 'photo_label': 1})
    for file_path in train_fake_files:
        train_final_json.append({'photo_path': file_path, 'photo_label': 0})
        all_final_json.append({'photo_path': file_path, 'photo_label': 0})
        fake_final_json.append({'photo_path': file_path, 'photo_label': 0})
        
    # Load test data
    test_real_files = glob.glob(test_real_path)
    test_fake_files = glob.glob(test_fake_path)
    for file_path in test_real_files:
        test_final_json.append({'photo_path': file_path, 'photo_label': 1})
        all_final_json.append({'photo_path': file_path, 'photo_label': 1})
        real_final_json.append({'photo_path': file_path, 'photo_label': 1})
    for file_path in test_fake_files:
        test_final_json.append({'photo_path': file_path, 'photo_label': 0})
        all_final_json.append({'photo_path': file_path, 'photo_label': 0})
        fake_final_json.append({'photo_path': file_path, 'photo_label': 0})
        
    # Print statistics
    print('\nLCC_FASD: ', len(all_final_json))
    print('LCC_FASD(train): ', len(train_final_json))
    print('LCC_FASD(test): ', len(test_final_json))
    print('LCC_FASD(all): ', len(all_final_json))
    print('LCC_FASD(real): ', len(real_final_json))
    print('LCC_FASD(fake): ', len(fake_final_json))

    # Write to JSON files
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()

def nuaa_process_cropped():
    train_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './nuua/'
    
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
        
    f_train = open(label_save_dir + 'train_label_crop.json', 'w')
    f_test = open(label_save_dir + 'test_label_crop.json', 'w')
    f_all = open(label_save_dir + 'all_label_crop.json', 'w')
    f_real = open(label_save_dir + 'real_label_crop.json', 'w')
    f_fake = open(label_save_dir + 'fake_label_crop.json', 'w')
    
    # Read train and test face lists from text files
    dataset_path = 'nuua_cropped/NUAA/Detectedface/'
    # client_train_file = os.path.join(dataset_path, 'client_train_face.txt')
    client_train_file = '/home/eeiith/Desktop/Project1/Kaustubh/IVP/SSDG-CVPR2020/data_label/Dataset/NUAA/raw/client_train_raw.txt'
    client_test_file = '/home/eeiith/Desktop/Project1/Kaustubh/IVP/SSDG-CVPR2020/data_label/Dataset/NUAA/raw/client_test_raw.txt'
    imposter_train_file = '/home/eeiith/Desktop/Project1/Kaustubh/IVP/SSDG-CVPR2020/data_label/Dataset/NUAA/raw/imposter_train_raw.txt'
    imposter_test_file = '/home/eeiith/Desktop/Project1/Kaustubh/IVP/SSDG-CVPR2020/data_label/Dataset/NUAA/raw/imposter_test_raw.txt'
    # client_test_file = os.path.join(dataset_path, 'client_test_face.txt')
    # imposter_train_file = os.path.join(dataset_path, 'imposter_train_face.txt')
    # imposter_test_file = os.path.join(dataset_path, 'imposter_test_face.txt')
    
    # Load train data
    with open(client_train_file, 'r') as f:
        client_train_faces = f.read().splitlines()
    with open(imposter_train_file, 'r') as f:
        imposter_train_faces = f.read().splitlines()
    
    client_train_faces = [i.replace("\\", "/") for i in client_train_faces]
    imposter_train_faces = [i.replace("\\", "/") for i in imposter_train_faces]
        
    for face in client_train_faces:
        file_path = os.path.join(dataset_path, 'ClientFace', face)
        
        train_final_json.append({'photo_path': file_path, 'photo_label': 1})
        all_final_json.append({'photo_path': file_path, 'photo_label': 1})
        real_final_json.append({'photo_path': file_path, 'photo_label': 1})
        
    for face in imposter_train_faces:
        file_path = os.path.join(dataset_path, 'ImposterFace', face)
        
        train_final_json.append({'photo_path': file_path, 'photo_label': 0})
        all_final_json.append({'photo_path': file_path, 'photo_label': 0})
        fake_final_json.append({'photo_path': file_path, 'photo_label': 0})
        
    # Load test data
    with open(client_test_file, 'r') as f:
        client_test_faces = f.read().splitlines()
    with open(imposter_test_file, 'r') as f:
        imposter_test_faces = f.read().splitlines()
        
    client_test_faces = [i.replace("\\", "/") for i in client_test_faces]
    imposter_test_faces = [i.replace("\\", "/") for i in imposter_test_faces]
      
    for face in client_test_faces:
        file_path = os.path.join(dataset_path, 'ClientFace', face)
        test_final_json.append({'photo_path': file_path, 'photo_label': 1})
        all_final_json.append({'photo_path': file_path, 'photo_label': 1})
        real_final_json.append({'photo_path': file_path, 'photo_label': 1})
    for face in imposter_test_faces:
        file_path = os.path.join(dataset_path, 'ImposterFace', face)
        test_final_json.append({'photo_path': file_path, 'photo_label': 0})
        all_final_json.append({'photo_path': file_path, 'photo_label': 0})
        fake_final_json.append({'photo_path': file_path, 'photo_label': 0})
    
    # Print statistics
    print('\nNUAA Detectedface: ', len(all_final_json))
    print('NUAA Detectedface(train): ', len(train_final_json))
    print('NUAA Detectedface(test): ', len(test_final_json))
    print('NUAA Detectedface(all): ', len(all_final_json))
    print('NUAA Detectedface(real): ', len(real_final_json))
    print('NUAA Detectedface(fake): ', len(fake_final_json))
    
    # Write to JSON files
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()

def replay_attack_process_cropped():
    train_final_json = []
    valid_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './replay_attack/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label_crop.json', 'w')
    f_valid = open(label_save_dir + 'valid_label_crop.json', 'w')
    f_test = open(label_save_dir + 'test_label_crop.json', 'w')
    f_all = open(label_save_dir + 'all_label_crop.json', 'w')
    f_real = open(label_save_dir + 'real_label_crop.json', 'w')
    f_fake = open(label_save_dir + 'fake_label_crop.json', 'w')
    # Open all_label.json file
    with open('./replay_attack/all_label.json', 'r') as f:
        replay_attack_data = json.load(f)
    # Change the path to the cropped images
    for i in range(len(replay_attack_data)):
        replay_attack_data[i]['photo_path'] = replay_attack_data[i]['photo_path'].replace('frames', 'replay_attack_cropped')
    # Save the cropped images to a new JSON file
    json.dump(replay_attack_data, f_all, indent=4)
    f_all.close()
    # Open train_label.json file
    with open('./replay_attack/train_label.json', 'r') as f:
        replay_attack_data = json.load(f)
    # Change the path to the cropped images
    for i in range(len(replay_attack_data)):
        replay_attack_data[i]['photo_path'] = replay_attack_data[i]['photo_path'].replace('frames', 'replay_attack_cropped')
    # Save the cropped images to a new JSON file
    json.dump(replay_attack_data, f_train, indent=4)    
    f_train.close()
    # Open valid_label.json file
    with open('./replay_attack/valid_label.json', 'r') as f:
        replay_attack_data = json.load(f)
    # Change the path to the cropped images
    for i in range(len(replay_attack_data)):
        replay_attack_data[i]['photo_path'] = replay_attack_data[i]['photo_path'].replace('frames', 'replay_attack_cropped')
    # Save the cropped images to a new JSON file
    json.dump(replay_attack_data, f_valid, indent=4)
    f_valid.close()
    # Open test_label.json file
    with open('./replay_attack/test_label.json', 'r') as f:
        replay_attack_data = json.load(f)
    # Change the path to the cropped images
    for i in range(len(replay_attack_data)):
        replay_attack_data[i]['photo_path'] = replay_attack_data[i]['photo_path'].replace('frames', 'replay_attack_cropped')
    # Save the cropped images to a new JSON file
    json.dump(replay_attack_data, f_test, indent=4)
    f_test.close()
    # Open real_label.json file
    with open('./replay_attack/real_label.json', 'r') as f:
        replay_attack_data = json.load(f)
    # Change the path to the cropped images
    for i in range(len(replay_attack_data)):
        replay_attack_data[i]['photo_path'] = replay_attack_data[i]['photo_path'].replace('frames', 'replay_attack_cropped')
    # Save the cropped images to a new JSON file
    json.dump(replay_attack_data, f_real, indent=4)
    f_real.close()
    # Open fake_label.json file
    with open('./replay_attack/fake_label.json', 'r') as f:
        replay_attack_data = json.load(f)
    # Change the path to the cropped images
    for i in range(len(replay_attack_data)):
        replay_attack_data[i]['photo_path'] = replay_attack_data[i]['photo_path'].replace('frames', 'replay_attack_cropped')
    # Save the cropped images to a new JSON file
    json.dump(replay_attack_data, f_fake, indent=4)
    f_fake.close()
    
    # Print statistics
    print('\nReplay Attack: ', len(replay_attack_data))
    print('Replay Attack(train): ', len(replay_attack_data))
    print('Replay Attack(valid): ', len(replay_attack_data))
    print('Replay Attack(test): ', len(replay_attack_data))
    print('Replay Attack(all): ', len(replay_attack_data))
    print('Replay Attack(real): ', len(replay_attack_data))
    print('Replay Attack(fake): ', len(replay_attack_data))

    
if __name__=="__main__":
    # Load LCC_FASD data
    lcc_fasd_process()
    nuaa_process()
    replay_attack_process()
    save_cropped_images()
    lcc_fasd_process_cropped()
    nuaa_process_cropped()
    replay_attack_process_cropped()