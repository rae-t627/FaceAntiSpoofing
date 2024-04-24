import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2
from utils.lbp import LBP

def equalise_image(img):
    img = np.array(img)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV color space
    h, s, v = cv2.split(img_hsv)  # Split channels\
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))

    v_eq = clahe.apply(v) 
    img_hsv_eq = cv2.merge((h, s, v_eq))  # Merge channels back
    img_eq = cv2.cvtColor(img_hsv_eq, cv2.COLOR_HSV2RGB)
    
    return img_eq
    
def crop_face(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    face_classifier = cv2.CascadeClassifier('/home/eeiith/Desktop/Project1/Kaustubh/IVP/SSDG-CVPR2020/utils/haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return image    
    x, y, w, h = faces[0]
    cropped_face = image.crop((x - 10, y - 10, x+w + 10, y+h + 10))
    return cropped_face
class YunpeiDataset(Dataset):
    def __init__(self, data_pd, transforms=None, train=True, equalize_hist=False, withLBP=False):
        self.equalize_hist = equalize_hist
        self.train = train
        self.withLBP = withLBP
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.Resize((256, 256)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.Resize((256, 256)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        root_path = "../../data_label/"
        if self.train:
            img_path = os.path.join(root_path, str(self.photo_path[item]))
            label = self.photo_label[item]
            img = Image.open(img_path)
            
            if self.equalize_hist:
                img = equalise_image(img)
                img = Image.fromarray(img)
                
            if self.withLBP:
                # Apply lbp on all three HSV channels and combine it into one image
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
                img_h = LBP(img[:, :, 0], 8, 1)
                img_s = LBP(img[:, :, 1], 8, 1)
                img_v = LBP(img[:, :, 2], 8, 1)
                
                img = np.dstack((img_h, img_s, img_v))
                img = Image.fromarray(img)
            
            img = self.transforms(img)
            label = torch.tensor(label)
            return img, label
        else:
            img_path = os.path.join(root_path, str(self.photo_path[item]))
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            img = Image.open(img_path)
            if self.equalize_hist:
                img = equalise_image(img)
                img = Image.fromarray(img)
                
            if self.withLBP:
                # Apply lbp on all three HSV channels and combine it into one image
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
                img_h = LBP(img[:, :, 0], 8, 1)
                img_s = LBP(img[:, :, 1], 8, 1)
                img_v = LBP(img[:, :, 2], 8, 1)
                img = np.dstack((img_h, img_s, img_v))
                img = Image.fromarray(img)
                
            img = self.transforms(img)
            label = torch.tensor(label)
            return img, label, videoID
