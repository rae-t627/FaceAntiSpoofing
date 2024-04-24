import sys
sys.path.append('../../')
from models.DGFAS import DG_model
import torch
from config import config
from utils.get_loader import get_dataset_modified

model = DG_model(config.model).cuda()

net_ = torch.load('/home/eeiith/Desktop/Project1/Kaustubh/IVP/SSDG-CVPR2020/experiment/backbone_classifier/L_R_N/best_model/model_best_0.12222_2.pth.tar')
model.load_state_dict(net_["state_dict"])
model.eval()

batch_size = 75

# load data
src1_train_dataloader_fake, src1_train_dataloader_real, \
src2_train_dataloader_fake, src2_train_dataloader_real, \
src3_train_dataloader_fake, src3_train_dataloader_real = get_dataset_modified(config.src1_data, config.src1_train_num_frames, 
                                    config.src2_data, config.src2_train_num_frames, 
                                    config.tgt_data, config.tgt_test_num_frames, batch_size)

src1_train_iter_real = iter(src1_train_dataloader_real)
src1_iter_per_epoch_real = len(src1_train_iter_real)
src2_train_iter_real = iter(src2_train_dataloader_real)
src2_iter_per_epoch_real = len(src2_train_iter_real)
src3_train_iter_real = iter(src3_train_dataloader_real)
src3_iter_per_epoch_real = len(src3_train_iter_real)

src1_train_iter_fake = iter(src1_train_dataloader_fake)
src1_iter_per_epoch_fake = len(src1_train_iter_fake)
src2_train_iter_fake = iter(src2_train_dataloader_fake)
src2_iter_per_epoch_fake = len(src2_train_iter_fake)
src3_train_iter_fake = iter(src3_train_dataloader_fake)
src3_iter_per_epoch_fake = len(src3_train_iter_fake)

src1_img_real, src1_label_real = next(src1_train_iter_real)
src1_img_real = src1_img_real.cuda()
src1_label_real = src1_label_real.cuda()
input1_real_shape = src1_img_real.shape[0]

src2_img_real, src2_label_real = next(src2_train_iter_real)
src2_img_real = src2_img_real.cuda()
src2_label_real = src2_label_real.cuda()
input2_real_shape = src2_img_real.shape[0]

src3_img_real, src3_label_real = next(src3_train_iter_real)
src3_img_real = src3_img_real.cuda()
src3_label_real = src3_label_real.cuda()
input3_real_shape = src3_img_real.shape[0]

src1_img_fake, src1_label_fake = next(src1_train_iter_fake)
src1_img_fake = src1_img_fake.cuda()
src1_label_fake = src1_label_fake.cuda()
input1_fake_shape = src1_img_fake.shape[0]

src2_img_fake, src2_label_fake = next(src2_train_iter_fake)
src2_img_fake = src2_img_fake.cuda()
src2_label_fake = src2_label_fake.cuda()
input2_fake_shape = src2_img_fake.shape[0]

src3_img_fake, src3_label_fake = next(src3_train_iter_fake)
src3_img_fake = src3_img_fake.cuda()
src3_label_fake = src3_label_fake.cuda()
input3_fake_shape = src3_img_fake.shape[0]

input_data = torch.cat([src1_img_real, src1_img_fake, src2_img_real, src2_img_fake, src3_img_real, src3_img_fake], dim=0)

source_label = torch.cat([src1_label_real, src1_label_fake,
                            src2_label_real, src2_label_fake, src3_label_real, src3_label_fake], dim=0)

######### forward #########
classifier_label_out, feature = model(input_data, config.norm_flag)

# using tsne to visualize the feature
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

tsne = TSNE(n_components=2, random_state=42)
feature = feature.cpu().detach().numpy()
feature = tsne.fit_transform(feature)

# 3D visualization
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# color = ['r', 'g', 'b', 'y', 'c', 'm']
# marker = ['o', 'x', 'o', 'x', 'o', 'x']
# for i in range(6):
#     ax.scatter(feature[i * batch_size:(i + 1) * batch_size, 0], feature[i * batch_size:(i + 1) * batch_size, 1], c=color[i], marker=marker[i], s = 15)
# plt.legend(['src1_real', 'src1_fake', 'src2_real', 'src2_fake'])
# plt.savefig("T-SNE plot")

plt.figure(figsize=(5, 5))
color = ['r', 'g', 'b', 'y', 'c', 'm']
marker = ['o', 'x', 'o', 'x', 'o', 'x']

for i in range(6):
    plt.scatter(feature[i * batch_size:(i + 1) * batch_size, 0], feature[i * batch_size:(i + 1) * batch_size, 1], c=color[i], marker=marker[i], s = 15)

plt.legend([config.src1_data + '_real', config.src1_data + '_fake', config.src2_data + '_real', config.src2_data + '_fake', config.tgt_data + '_real', config.tgt_data + '_fake'])
plt.title(config.src1_data + ' and ' + config.src2_data + ' to ' + config.tgt_data + ' T-SNE plot')
plt.savefig("T-SNE plot")