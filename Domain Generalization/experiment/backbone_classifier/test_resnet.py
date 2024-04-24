import sys
sys.path.append('../../')
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from config import config
from utils.utils import get_points
from utils.dataset import YunpeiDataset
from utils.utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, calculate_threshold
from sklearn.metrics import confusion_matrix
from models.DGFAS import DG_model
import matplotlib.pyplot as plt
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

def test(test_dataloader, model, threshold):
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    number = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for iter, (input, target, videoID) in enumerate(test_dataloader):
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            cls_out, _ = model(input, config.norm_flag)
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()
            correct += (cls_out.argmax(dim=1) == target).sum().item()
            total += target.size(0)
            for i in range(len(prob)):
                if (videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                    number += 1
                    if (number % 100 == 0):
                        print('**Testing** ', number, ' photos done!')
    print('**Testing** ', number, ' photos done!')
    
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        
        avg_single_video_output = sum(output_dict_tmp[key]) / len(output_dict_tmp[key])
        avg_single_video_target = sum(target_dict_tmp[key]) / len(target_dict_tmp[key])
        
        
        if avg_single_video_prob > 0.5 :
            avg_single_video_output = torch.tensor([-avg_single_video_prob, avg_single_video_prob]).view(1, 2)
        else:
            avg_single_video_output = torch.tensor([avg_single_video_prob, -avg_single_video_prob]).view(1, 2)
        
        avg_single_video_output = avg_single_video_output.cuda()
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
        valid_top1.update(acc_valid[0])
        

    cur_EER_valid, threshold, FRR_list, FAR_list = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    
    acc = correct*100 / total
    fpr, tpr, _ = metrics.roc_curve(label_list,  prob_list)
    auc_score = metrics.roc_auc_score(label_list, prob_list)
    
    data = np.column_stack((fpr, tpr)) 
    save_path_roc = os.path.join('save_results', config.tgt_data)
    if not os.path.exists(save_path_roc):
        os.makedirs(save_path_roc)
    np.savetxt(os.path.join(save_path_roc, 'roc_curve.csv'), data, delimiter=',', header='fpr,tpr', comments='')
    
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(save_path_roc, 'roc.png'))
    
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in prob_list]
    conf_matrix = confusion_matrix(label_list, y_pred_binary)

    TN, FP, FN, TP = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]

    FAR = FP / (FP + TN)
    FRR = FN / (FN + TP)

    HTER = (FAR + FRR) / 2
    
    return [valid_top1.avg, cur_EER_valid, HTER, auc_score, ACC_threshold, threshold]

def main():
    net = DG_model(config.model).cuda()
    test_data = get_points(config.tgt_data, config.tgt_test_num_frames, config.flag_test)
    test_dataloader = DataLoader(YunpeiDataset(test_data, train=False), batch_size=1, shuffle=False)
    print('\n')
    print("**Testing** Get test files done!")
    
    # load model
    net_ = torch.load(config.best_model_path + config.tgt_best_model_name)
    net.load_state_dict(net_["state_dict"])
    threshold = net_["threshold"]
    
    # test model
    test_args = test(test_dataloader, net, threshold)
    print('\n===========Test Info===========\n')
    print(config.tgt_data, 'Test acc: %5.4f' %(test_args[0]))
    print(config.tgt_data, 'Test EER: %5.4f' %(test_args[1]))
    print(config.tgt_data, 'Test HTER: %5.4f' %(test_args[2]))
    print(config.tgt_data, 'Test AUC: %5.4f' % (test_args[3]))
    print(config.tgt_data, 'Test ACC_threshold: %5.4f' % (test_args[4]))
    print('\n===============================\n')

if __name__ == '__main__':
    main()
