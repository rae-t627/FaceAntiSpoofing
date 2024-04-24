import sys
sys.path.append('../../')

from models.DGFAS import Feature_Generator_ResNet18, Feature_Embedder_ResNet18
from utils.get_loader import get_dataset
from utils.utils import save_checkpoint
from config import config
import numpy as np
import torch
import torch.nn as nn
from utils.evaluate import eval2
from sklearn.svm import SVC
import os

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag):
        if(norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.backbone = Feature_Generator_ResNet18()
        self.embedder = Feature_Embedder_ResNet18()
        self.classifier = Classifier()
        
    def forward(self, x, norm_flag):
        x = self.backbone(x)
        features = self.embedder(x, norm_flag)
        cls = self.classifier(features, norm_flag)
        return cls, features

src1_train_dataloader_fake, src1_train_dataloader_real, \
src2_train_dataloader_fake, src2_train_dataloader_real, \
tgt_valid_dataloader = get_dataset(config.src1_data, config.src1_train_num_frames, 
                                    config.src2_data, config.src2_train_num_frames, 
                                    config.tgt_data, config.tgt_test_num_frames, 1)

best_model_ACC = 0.0
best_model_HTER = 1.0
best_model_ACER = 1.0
best_model_AUC = 0.0

# create paths:
os.makedirs(config.checkpoint_path, exist_ok=True)
os.makedirs(config.best_model_path, exist_ok=True)

# Create an iterator for the dataloader
src1_train_iter_real = iter(src1_train_dataloader_real)
src1_iter_per_epoch_real = len(src1_train_iter_real)
src2_train_iter_real = iter(src2_train_dataloader_real)
src2_iter_per_epoch_real = len(src2_train_iter_real)
src1_train_iter_fake = iter(src1_train_dataloader_fake)
src1_iter_per_epoch_fake = len(src1_train_iter_fake)
src2_train_iter_fake = iter(src2_train_dataloader_fake)
src2_iter_per_epoch_fake = len(src2_train_iter_fake)

# Define model
model = model().cuda()

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

max_iter = config.max_iter
prev_loss = 10000
iter_per_epoch = 10
epoch = 1

for iter_num in range(max_iter+1):
    if (iter_num % src1_iter_per_epoch_real == 0):
        src1_train_iter_real = iter(src1_train_dataloader_real)
    if (iter_num % src2_iter_per_epoch_real == 0):
        src2_train_iter_real = iter(src2_train_dataloader_real)
    if (iter_num % src1_iter_per_epoch_fake == 0):
        src1_train_iter_fake = iter(src1_train_dataloader_fake)
    if (iter_num % src2_iter_per_epoch_fake == 0):
        src2_train_iter_fake = iter(src2_train_dataloader_fake)
    if (iter_num != 0 and iter_num % iter_per_epoch == 0):
        epoch = epoch + 1

    model.train(True)
    optimizer.zero_grad()
    
    ######### data prepare #########
    src1_img_real, src1_label_real = next(src1_train_iter_real)
    src1_img_real = src1_img_real.cuda()
    src1_label_real = src1_label_real.cuda()
    input1_real_shape = src1_img_real.shape[0]

    src2_img_real, src2_label_real = next(src2_train_iter_real)
    src2_img_real = src2_img_real.cuda()
    src2_label_real = src2_label_real.cuda()
    input2_real_shape = src2_img_real.shape[0]

    src1_img_fake, src1_label_fake = next(src1_train_iter_fake)
    src1_img_fake = src1_img_fake.cuda()
    src1_label_fake = src1_label_fake.cuda()
    input1_fake_shape = src1_img_fake.shape[0]

    src2_img_fake, src2_label_fake = next(src2_train_iter_fake)
    src2_img_fake = src2_img_fake.cuda()
    src2_label_fake = src2_label_fake.cuda()
    input2_fake_shape = src2_img_fake.shape[0]

    input_data = torch.cat([src1_img_real, src1_img_fake, src2_img_real, src2_img_fake], dim=0)

    source_label = torch.cat([src1_label_real, src1_label_fake,
                                src2_label_real, src2_label_fake], dim=0)

    ######### forward #########
    classifier_label_out, feature = model(input_data, config.norm_flag)
    
    loss = criterion(classifier_label_out, source_label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if (iter_num != 0 and (iter_num+1) % iter_per_epoch == 0):
        # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:AUC, 5:threshold, 6:ACC_threshold
        valid_args = eval2(tgt_valid_dataloader, model, config.norm_flag)
        
        # judge model according to HTER
        is_best = valid_args[3] < best_model_HTER
        best_model_HTER = min(valid_args[3], best_model_HTER)
        threshold = valid_args[5]
        if (valid_args[3] <= best_model_HTER):
            best_model_ACC = valid_args[6]
            best_model_AUC = valid_args[4]

        save_list = [epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER, threshold]
        save_checkpoint(save_list, is_best, model, optimizer, config.gpus, config.checkpoint_path, config.best_model_path)
        
        
        # print(type(valid_args[0]), type(valid_args[1]), type(valid_args[2]), type(valid_args[3]), type(valid_args[4]), type(valid_args[5]), type(valid_args[6]))
        
        print('Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}, EER: {:.4f}, HTER: {:.4f}, AUC: {:.4f}, Threshold: {:.4f}, ACC_threshold: {:.4f}'.format(
            epoch, valid_args[0], valid_args[7], valid_args[2], best_model_HTER, valid_args[4], valid_args[5], valid_args[6]))
        
        # print("Epoch:{}, Loss: {}, HTER: {}".format(epoch, valid_args[0], valid_args[3]))
        
