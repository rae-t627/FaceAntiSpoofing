class DefaultConfigs(object):
    seed = 666
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    init_lr = 0.01
    lr_epoch_1 = 0
    lr_epoch_2 = 100
    # model
    pretrained = True
    model = 'resnet18'     # resnet18 or maddg
    # training parameters
    gpus = '0, 1'
    batch_size = 10
    norm_flag = True
    max_iter = 2000
    lambda_triplet = 2
    lambda_adreal = 0.1
    # test model name
    tgt_best_model_name = 'model_best_0.04_20.pth.tar' 
    # source data information
    src1_data = 'lcc_fasd'
    src1_train_num_frames = 1000
    src2_data = 'nuua'
    src2_train_num_frames = 1000
    # target data information
    tgt_data = 'lcc_fasd'
    tgt_test_num_frames = 2000
    flag_test = 2
    
    # paths information
    checkpoint_path = './' + 'replay_attack' + '_checkpoint/' + model + '/DGFANet/'
    best_model_path = './' + 'replay_attack' + '_checkpoint/' + model + '/best_model/'
    logs = './logs/'

config = DefaultConfigs()
