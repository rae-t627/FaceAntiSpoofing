import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2

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
    def __init__(self, data_pd, transforms=None, train=True):
        self.train = train
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
            img = self.transforms(img)
            label = torch.tensor(label)
            return img, label
        else:
            img_path = os.path.join(root_path, str(self.photo_path[item]))
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            img = Image.open(img_path)
            img = self.transforms(img)
            label = torch.tensor(label)
            return img, label, videoID
