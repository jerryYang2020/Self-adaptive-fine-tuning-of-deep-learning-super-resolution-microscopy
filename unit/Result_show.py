import cv2
import torch
import numpy as np
from torch.autograd import Variable
from skimage import io,filters,transform
from torchvision.transforms import ToTensor
import os


def ReconSR(model,bin,predict_size,step,gap,imagedata_dir,imageedge_dir,savedir,FilenameSave,imagetypeSave):

    FilenameSave2 = ((imagedata_dir.split('\\')[-1]).split('.'))[0]+'_SRN_fine-tuning'
    savedir = os.path.join(savedir, FilenameSave2)
    os.makedirs(savedir, exist_ok=True)


    iterator = 10000

    image = io.imread(imagedata_dir)
    edge = io.imread(imageedge_dir)

    if len(image.shape)==3:
        image = image[0, :, :]
        edge = edge[0, :, :]

    img_size = [image.shape[0], image.shape[1]]



    model4x = model
    model4x = model4x.eval()



    coorx = 0
    corry = 0
    linebreak = 0
    lastline = 0
    input_imageBin = image / image.max()
    edge = edge/edge.max()

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

        input=torch.cat((input2,input1),1)




        out = model4x(input)

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



    input_imagepredict3=input_imagepredict3.astype('float32')
    input_imagepredictJPGE = input_imagepredict3
    input_imagepredict3=input_imagepredict3/input_imagepredict3.max()
    input_imagepredictJPGE = input_imagepredictJPGE / input_imagepredictJPGE.max()
    input_imagepredictJPGE = input_imagepredictJPGE * 255 - 0.00001
    input_imagepredictJPGE = input_imagepredictJPGE.astype(np.uint8)
    cv2.imwrite(os.path.join(savedir, FilenameSave  + '.jpg'), input_imagepredictJPGE)
    cv2.imwrite(os.path.join(savedir, FilenameSave  + imagetypeSave), input_imagepredict3)
    return FilenameSave2







