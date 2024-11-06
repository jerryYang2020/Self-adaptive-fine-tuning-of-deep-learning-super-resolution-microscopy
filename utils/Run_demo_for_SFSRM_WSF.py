from options.get_options import get_options
import argparse
from utils.SuperResolutionRecon import SuperResolutionRecon
from utils.Data_from_image import get_training_set_SFSRM_SMLM
from utils.WSF import SRmodel_WSF
from shutil import rmtree
import os
from natsort import natsorted
import torch


 

def runFineTuning(
        lrImgPath,
        edgeImgPath,
        SMLMImgPath,
        datadir_save = 'test_data\SEC61B_WSF',
        SFSRMmodeldir = 'Pretrained_model\SFSRM_SRmodel_SEC61B.pth',
        scale = 8,
        timelen_WSF = 10,
        iteration_times_WSF = 3,
        SRlr_WSF = 0.00001,
        step = 40,
        gap = 2,
        trainsize = [64, 64],
        predict_size = [64, 64]
):
    
    scale = int(scale)
    timelen_WSF = int(timelen_WSF)
    iteration_times_WSF = int(iteration_times_WSF)
    SRlr_WSF = float(SRlr_WSF)
    step = int(step)
    gap = int(gap)



    print(lrImgPath, edgeImgPath, SMLMImgPath)
    if os.path.exists(datadir_save):
        rmtree(datadir_save)
    train_set = get_training_set_SFSRM_SMLM(datadir_save, lrImgPath, edgeImgPath, trainsize, SMLMImgPath, scale)
    model, fineTuningDir = SRmodel_WSF(train_set,lrImgPath, edgeImgPath, datadir_save, scale, iteration_times_WSF, SRlr_WSF, timelen_WSF, SFSRMmodeldir, predict_size, step, gap)

    file_list = natsorted(os.listdir(fineTuningDir), key=lambda y: y.lower())
    file_list = [os.path.join(fineTuningDir, path) for path in file_list if '.tif' not in path]
    return file_list

def runSuperResolutionRecon(
        index,
        lrImgPath,
        edgeImgPath,
        SMLMImgPath,
        datadir_save = 'test_data\SEC61B_WSF',
        SFSRMmodeldir = 'Pretrained_model\SFSRM_SRmodel_SEC61B',
        scale = 8,
        trainsize = [64, 64],
        timelen_WSF = 10,
        iteration_times_WSF = 3,
        SRlr_WSF = 0.00001,
        predict_size = [64, 64],
        step = 40,
        gap = 2
):
    modelpath = os.path.join(datadir_save, 'Fine-tuning_Models\\'+str(int(index))+'.pth')
    model = torch.load(modelpath)
    output_path = SuperResolutionRecon(model, lrImgPath, edgeImgPath, datadir_save, scale, predict_size, step, gap)
    return output_path

if __name__ == '__main__':
    runFineTuning()
    print("===> Successful runing SFSRM-WSF: "
          "All data is available in preset savedir")

