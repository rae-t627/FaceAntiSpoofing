import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.dataset import YunpeiDataset
from utils.utils import get_points

def get_dataset(src1_data, src1_train_points, src2_data, src2_train_points,
                tgt_data, tgt_test_points, batch_size):
    # Load the Source and Target data. Get src1_train_points and src2_train_points datapoints from the source data and tgt_test_points datapoints from the target data.
    print('Load Source Data')
    print('Source Data: ', src1_data)
    src1_train_data_fake = get_points(src1_data, src1_train_points, 0)
    src1_train_data_real = get_points(src1_data, src1_train_points, 1)      
    print('Source Data: ', src2_data)
    src2_train_data_fake = get_points(src2_data, src2_train_points, 0)
    src2_train_data_real = get_points(src2_data, src2_train_points, 1)

    print('Load Target Data')
    print('Target Data: ', tgt_data)
    tgt_test_data = get_points(tgt_data, tgt_test_points, 2)
#     tgt_test_data = sample_frames(flag=2, num_frames=tgt_test_points, dataset_name=tgt_data)

    src1_train_dataloader_fake = DataLoader(YunpeiDataset(src1_train_data_fake, train=True),
                                            batch_size=batch_size, shuffle=True)
    src1_train_dataloader_real = DataLoader(YunpeiDataset(src1_train_data_real, train=True),
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_fake = DataLoader(YunpeiDataset(src2_train_data_fake, train=True),
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_real = DataLoader(YunpeiDataset(src2_train_data_real, train=True),
                                            batch_size=batch_size, shuffle=True)
    tgt_dataloader = DataLoader(YunpeiDataset(tgt_test_data, train=False), batch_size=batch_size, shuffle=False)
    
    return src1_train_dataloader_fake, src1_train_dataloader_real, \
           src2_train_dataloader_fake, src2_train_dataloader_real, \
           tgt_dataloader
           
def get_dataset_modified(src1_data, src1_train_points, src2_data, src2_train_points,
                src3_data, src3_train_points, batch_size):
    # Load the Source and Target data. Get src1_train_points and src2_train_points datapoints from the source data and tgt_test_points datapoints from the target data.
    print('Load Source Data')
    print('Source Data: ', src1_data)
    src1_train_data_fake = get_points(src1_data, src1_train_points, 0)
    src1_train_data_real = get_points(src1_data, src1_train_points, 1)      
    print('Source Data: ', src2_data)
    src2_train_data_fake = get_points(src2_data, src2_train_points, 0)
    src2_train_data_real = get_points(src2_data, src2_train_points, 1)
    print('Source Data: ', src3_data)
    src3_train_data_fake = get_points(src3_data, src3_train_points, 0)
    src3_train_data_real = get_points(src3_data, src3_train_points, 1)

    src1_train_dataloader_fake = DataLoader(YunpeiDataset(src1_train_data_fake, train=True),
                                            batch_size=batch_size, shuffle=True)
    src1_train_dataloader_real = DataLoader(YunpeiDataset(src1_train_data_real, train=True),
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_fake = DataLoader(YunpeiDataset(src2_train_data_fake, train=True),
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_real = DataLoader(YunpeiDataset(src2_train_data_real, train=True),
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_fake = DataLoader(YunpeiDataset(src3_train_data_fake, train=True),
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_real = DataLoader(YunpeiDataset(src3_train_data_real, train=True),
                                            batch_size=batch_size, shuffle=True)

    return src1_train_dataloader_fake, src1_train_dataloader_real, \
           src2_train_dataloader_fake, src2_train_dataloader_real, \
           src3_train_dataloader_fake, src3_train_dataloader_real