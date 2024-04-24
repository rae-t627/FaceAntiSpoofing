import json
import math
import re
import pandas as pd
import torch
import os
import sys
import shutil
import matplotlib.pyplot as plt

def adjust_learning_rate(optimizer, epoch, init_param_lr, lr_epoch_1, lr_epoch_2):
    '''
    Sets the learning rate to the initial LR decayed by 10 every lr_epoch_1 and lr_epoch_2 epochs
    '''
    i = 0
    for param_group in optimizer.param_groups:
        init_lr = init_param_lr[i]
        i += 1
        if(epoch <= lr_epoch_1):
            param_group['lr'] = init_lr * 0.1 ** 0
        elif(epoch <= lr_epoch_2):
            param_group['lr'] = init_lr * 0.1 ** 1
        else:
            param_group['lr'] = init_lr * 0.1 ** 2

def draw_roc(
    frr_list, far_list, roc_auc):
    '''
        Draw the ROC curve
    '''
    plt.switch_backend('agg')
    plt.rcParams['figure.figsize'] = (6.0, 6.0)
    plt.title('ROC')
    plt.plot(far_list, frr_list, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='upper right')
    plt.plot([0, 1], [1, 0], 'r--')
    plt.grid(ls='--')
    plt.ylabel('False Negative Rate')
    plt.xlabel('False Positive Rate')
    save_dir = './save_results/ROC/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig('./save_results/ROC/ROC.png')
    file = open('./save_results/ROC/FAR_FRR.txt', 'w')
    save_json = []
    dict = {}
    dict['FAR'] = far_list
    dict['FRR'] = frr_list
    save_json.append(dict)
    json.dump(save_json, file, indent=4)
    
def extract_id_number(image_name):
    '''
        extract the id number from the image name
    '''
    pattern = r'id(\d+)'
    match = re.search(pattern, image_name)
    if match:
        id_number = match.group(1)
        return int(id_number)
    else:
        return None
    
def extract_client_number(image_name):
    '''
        extract the client number from the image name
    '''
    pattern = r'client(\d+)'
    match = re.search(pattern, image_name)
    if match:
        client_number = match.group(1)
        return int(client_number)
    else:
        return None

def get_points(dataset_name, num_points, flag):
    '''
        get the dataset with num_points from dataset_name
        return: the dataset's path and label
    '''
    root_path = '../../data_label/' + dataset_name
    if(flag == 0): # select the fake images
        label_path = root_path + '/fake_label_crop.json'
        save_label_path = root_path + '/choose_fake_label_crop.json'
    elif(flag == 1): # select the real images
        label_path = root_path + '/real_label_crop.json'
        save_label_path = root_path + '/choose_real_label_crop.json'
    elif(flag == 2): # select the test images
        label_path = root_path + '/test_label_crop.json'
        save_label_path = root_path + '/choose_test_label_crop.json'
    else: # select all the real and fake images
        label_path = root_path + '/all_label_crop.json'
        save_label_path = root_path + '/choose_all_label_crop.json'
    
    print("label_path: ", label_path)
    print("save_label_path: ", save_label_path)
    all_label_json = json.load(open(label_path, 'r'))
    length = len(all_label_json)

    final_json = []
    for i in range(length):
        dict = {}
        dict['photo_path'] = all_label_json[i]['photo_path']
        dict['photo_label'] = all_label_json[i]['photo_label']
        try:
            if dataset_name == 'lcc_fasd':
                image_name = all_label_json[i]['photo_path'].split('/')[-1]
                contents = image_name.split('_')
                if contents[0].isdigit():
                    dict['photo_belong_to_video_ID'] = int(contents[0])
                elif contents[0] == 'real' or contents[0] == 'spoof':
                    dict['photo_belong_to_video_ID'] = int(contents[1].split('.')[0])
                else:
                    dict['photo_belong_to_video_ID'] = extract_id_number(image_name)
                    
            elif dataset_name == 'replay_attack':
                image_name = all_label_json[i]['photo_path'].split('/')[-2]
                dict['photo_belong_to_video_ID'] = extract_client_number(image_name)
                
            elif dataset_name == 'nuaa':
                image_name = all_label_json[i]['photo_path'].split('/')[-2]
                dict['photo_belong_to_video_ID'] = int(image_name)
                
            # Add the photo label at the end of video ID
            dict['photo_belong_to_video_ID'] = str(dict['photo_belong_to_video_ID']) + '_' + str(dict['photo_label'])
            
        except:
            print("Error: ", all_label_json[i]['photo_path'])
            raise ValueError("Error: ", all_label_json[i]['photo_path'])
        final_json.append(dict)
        
    if (flag == 0):
        print("Total video number(fake): ", len(final_json), dataset_name)
    elif (flag == 1):
        print("Total video number(real): ", len(final_json), dataset_name)
    elif (flag == 2):                # contents = image_name.split('_')

        print("Total video number(test): ", len(final_json), dataset_name)
    else:
        print("Total video number(target): ", len(final_json), dataset_name)
        
    json.dump(final_json, open(save_label_path, 'w'), indent=4)
    f_json = open(save_label_path)
    sample_data_pd = pd.read_json(f_json)
    return sample_data_pd                                           

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # Get the value from target
        target = target.view(-1, 1)
        # Round target to the nearest integer
        target = target.round()        
        _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()
        correct = pred.eq(target.expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def mkdirs(checkpoint_path, best_model_path, logs):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(logs):
        os.mkdir(logs)

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)
    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def save_checkpoint(save_list, is_best, model, optimizer, gpus, checkpoint_path, best_model_path, filename='_checkpoint.pth.tar'):
    epoch = save_list[0]
    valid_args = save_list[1]
    best_model_HTER = round(save_list[2], 5)
    best_model_ACC = save_list[3]
    best_model_ACER = save_list[4]
    threshold = save_list[5]
    if(len(gpus) > 1):
        old_state_dict = model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            flag = k.find('module.')
            if (flag != -1):
                k = k.replace('module.', '')
            new_state_dict[k] = v
        
        state = {
            "epoch": epoch,
            "state_dict": new_state_dict,
            "optimizer" : optimizer.state_dict(),
            "valid_arg": valid_args,
            "best_model_EER": best_model_HTER,
            "best_model_ACER": best_model_ACER,
            "best_model_ACC": best_model_ACC,
            "threshold": threshold
        }
    else:
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "valid_arg": valid_args,
            "best_model_EER": best_model_HTER,
            "best_model_ACER": best_model_ACER,
            "best_model_ACC": best_model_ACC,
            "threshold": threshold
        }
    filepath = checkpoint_path + filename
    torch.save(state, filepath)
    # just save best model
    if is_best:
        shutil.copy(filepath, best_model_path + 'model_best_' + str(best_model_HTER) + '_' + str(epoch) + '.pth.tar')

def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()