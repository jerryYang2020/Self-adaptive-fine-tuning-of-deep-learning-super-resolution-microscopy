import argparse
from utils.Data_from_image import get_training_set
from utils.SAFT  import SRmodel_SSF
from utils.SuperResolutionRecon import SuperResolutionRecon
import os
from natsort import natsorted
import torch



def runSSFFineTuning(
        lrImgPath,
        edgeImgPath,
        datadir_save = 'test_data\SEC61B_SSF',
        SFSRMmodeldir = 'Pretrained_model\SFSRM_SRmodel_SEC61B.pth',
        DENmodeldir = 'Pretrained_model\Denoisingmodel_SEC61B.pth',

        iteration_times = 3,
        sigma = 1,
        ksize = 5,
        SRlr = 0.000001,
        continuity_level = 1,
        sparse_level = 1,
        consistency_level = 1,

        joint_Training = True, 
        trainsize = [64, 64],
        predict_size = [64, 64]
):
    
    scale = 8
    timelen = 5
    step = 40
    gap = 5

    iteration_times = int(iteration_times)
    sigma = int(sigma)
    ksize = int(ksize)
    SRlr = float(SRlr)
    continuity_level = int(continuity_level)
    sparse_level = int(sparse_level)
    consistency_level = int(consistency_level)


    print(lrImgPath, edgeImgPath)

    train_set = get_training_set(datadir_save, lrImgPath, edgeImgPath, trainsize, timelen)
    model, fineTuningDir = SRmodel_SSF(train_set,lrImgPath, edgeImgPath, datadir_save, SFSRMmodeldir, DENmodeldir, scale, iteration_times, sigma, ksize, SRlr, continuity_level, sparse_level, consistency_level, joint_Training, predict_size, step, gap)


    file_list = natsorted(os.listdir(fineTuningDir), key=lambda y: y.lower())
    file_list = [os.path.join(fineTuningDir, path) for path in file_list if '.tif' not in path]

    return file_list

def runSSFSuperResolutionRecon(
        index,
        lrImgPath,
        edgeImgPath,
        datadir_save = 'test_data\SEC61B_SSF',
        SFSRMmodeldir = 'Pretrained_model\SFSRM_SRmodel_SEC61B',
        scale = 8,
        trainsize = [64, 64],
        timelen_WSF = 10,
        iteration_times_WSF = 3,
        SRlr_WSF = 0.00001,
        predict_size = [64, 64],
        step = 40,
        gap = 5
):

    datadir_save_=os.path.join(datadir_save,'Fine-tuning_Models')
    modelpath = os.path.join(datadir_save_,str(int(index))+'.pth')
    model = torch.load(modelpath)
    output_path = SuperResolutionRecon(model, lrImgPath, edgeImgPath, datadir_save, scale, predict_size, step, gap)
    return output_path






