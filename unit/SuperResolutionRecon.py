import cv2
import torch
import numpy as np
from torch.autograd import Variable
from skimage import io
from torchvision.transforms import ToTensor
import os
from tqdm import tqdm
from . import gen_vedio
from shutil import rmtree


def SuperResolutionRecon(model, lrImgPath, edgeImgPath, datadir_save, scale, predict_size, step, gap,imagetypeSave='.tif'):

    imagedata_dir = lrImgPath
    imageedge_dir = edgeImgPath
    savedir = datadir_save
    bin = scale
    predict_size = predict_size
    step = step
    gap = gap

    print("===> Starting Super-resolution Reconstruction")

    FilenameSave = ((imagedata_dir.split('\\')[-1]).split('.'))[0]+'_SuperResolution'
    savedir = os.path.join(savedir, FilenameSave)
    if os.path.exists(savedir):
        rmtree(savedir)
        os.makedirs(savedir, exist_ok=True)
    else:   
        os.makedirs(savedir, exist_ok=True)

    iterator = 10000


    image = io.imread(imagedata_dir)
    edgemap = io.imread(imageedge_dir)

    if len(image.shape)==3:
        img_size = [image.shape[1], image.shape[2]]
        imagelen = image.shape[0]
    if len(image.shape)==2:
        img_size = [image.shape[0], image.shape[1]]
        imagelen = 1

    modelSR = model
    modelSR = modelSR.eval()

    for i in tqdm(range(imagelen)):

        coorx = 0
        corry = 0
        linebreak = 0
        lastline = 0

        if len(image.shape) == 3:
            input_imageBin = image[i, :, :] / image[i, :, :].max()
            edge = edgemap[i, :, :] / edgemap[i, :, :].max()
        if len(image.shape) == 2:
            input_imageBin = image / image.max()
            edge = edgemap/ edgemap.max()

        input_imagepredict3 = np.zeros((img_size[0] * bin, img_size[1] * bin))

        for b in range(iterator):
            if corry + predict_size[1] >= img_size[1]:
                corry = img_size[1] - 1 - predict_size[1]
                linebreak = 1

            input_image1 = input_imageBin[coorx:coorx + predict_size[0], corry:corry + predict_size[1]]
            input1 = ToTensor()(input_image1)
            input1 = Variable(input1.unsqueeze(0).float(), requires_grad=False)
            input1 = input1.cuda()

            input_image2 = edge[coorx:coorx + predict_size[0], corry:corry + predict_size[1]]
            input2 = ToTensor()(input_image2)
            input2 = Variable(input2.unsqueeze(0).float(), requires_grad=False)
            input2 = input2.cuda()

            input = torch.cat((input2, input1), 1)

            out = modelSR(input)

            out4x = out

            out4x = torch.abs(out4x)
            out4x = out4x.detach().cpu().numpy().squeeze()

            input_imagepredict3[coorx * bin + gap * bin:coorx * bin + predict_size[0] * bin - gap * bin,
            corry * bin + gap * bin:corry * bin + predict_size[1] * bin - gap * bin] \
                = out4x[gap * bin:predict_size[0] * bin - gap * bin, gap * bin:predict_size[1] * bin - gap * bin]

            if linebreak == 1 and lastline == 1:
                break
            if linebreak == 0:
                corry = corry + step
            if linebreak == 1:
                corry = 0
                coorx = coorx + step
                linebreak = 0
            if coorx + predict_size[0] >= img_size[0]:
                coorx = img_size[0] - 1 - predict_size[0]
                lastline = 1
            if linebreak == 1 and lastline == 1:
                break

        input_imagepredict3 = input_imagepredict3.astype('float32')

        cv2.imwrite(os.path.join(savedir, str(i + 1) + imagetypeSave), input_imagepredict3)
    jpg_path_name = os.path.join(datadir_save, "JPGE")
    output_folder = os.path.join(datadir_save, "mp")
    filename = os.path.join(datadir_save, FilenameSave)
    output_path = gen_vedio.merge_image_to_video(filename, output_format='mp4', jpg_folder = jpg_path_name, output_folder=output_folder)
    return output_path




