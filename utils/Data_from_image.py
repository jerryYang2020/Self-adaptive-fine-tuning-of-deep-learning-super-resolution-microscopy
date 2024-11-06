import torch
import os
import random
from os import listdir
from os.path import join
from skimage import io,transform
import torch.utils.data as data
from torchvision import transforms
import numpy as np





class DatasetFromFolder_SFSRM2(data.Dataset):
    def __init__(self, image_dir_1, input_transform=None, target_transform=None):
        super(DatasetFromFolder_SFSRM2, self).__init__()
        self.image_dir_1 = image_dir_1
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        input = io.imread(os.path.join(self.image_dir_1,str(index+1)+'_wf.tif'))
        edge = io.imread(os.path.join(self.image_dir_1,str(index+1)+'_eg.tif'))
        s = input.shape

        if len(input.shape)==3:
            input2 = np.zeros(shape=[s[1], s[2], s[0]])
            input2 = input2.astype('float32')
            for i in range(s[0]):
                input2[:, :, i] = input[i, :, :]
            edge2 = np.zeros(shape=[s[1], s[2], s[0]])
            edge2 = edge2.astype('float32')
            for i in range(s[0]):
                edge2[:, :, i] = edge[i, :, :]

        if len(input.shape)==2:
            input2=input
            edge2=edge


        input2 = self.transform(input2)
        edge2 = self.transform(edge2)

        return input2,edge2

    def __len__(self):
        return len(os.listdir(self.image_dir_1))//2

class DatasetFromFolder_SFSRM3(data.Dataset):
    def __init__(self, image_dir_1, input_transform=None, target_transform=None):
        super(DatasetFromFolder_SFSRM3, self).__init__()
        self.image_dir_1 = image_dir_1
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 转化为pytorch中的tensor
        ])

    def __getitem__(self, index):
        input = io.imread(os.path.join(self.image_dir_1,str(index+1)+'_wf.tif'))
        edge = io.imread(os.path.join(self.image_dir_1,str(index+1)+'_eg.tif'))
        smlm = io.imread(os.path.join(self.image_dir_1,str(index+1)+'_smlm.tif'))



        s = input.shape
        input2=np.zeros(shape=[s[1],s[2],s[0]])
        input2=input2.astype('float32')
        for i in range(s[0]):
            input2[:,:,i]=input[i,:,:]
        edge2=np.zeros(shape=[s[1],s[2],s[0]])
        edge2=edge2.astype('float32')
        for i in range(s[0]):
            edge2[:,:,i]=edge[i,:,:]


        input2 = self.transform(input2)
        edge2 = self.transform(edge2)
        smlm = self.transform(smlm)

        input2=input2.unsqueeze(1)
        edge2=edge2.unsqueeze(1)

        return [input2,edge2,smlm]

    def __len__(self):
        return len(os.listdir(self.image_dir_1))//3



def get_training_set(datadir_save, lrImgPath, edgeImgPath, trainsize, timelen):

    savedir = datadir_save
    datadir = lrImgPath
    edgedir = edgeImgPath
    cutsize = trainsize
    timelen = timelen

    savedir1=os.path.join(savedir,'Hessian_patches')
    os.makedirs(savedir1,exist_ok=True)
    get_traindata_SFSRM2(datadir,edgedir,savedir1,cutsize,timelen)

    print("===> Setting Creating training Data_for_SFSRM-SSF")

    return DatasetFromFolder_SFSRM2(savedir1)

def get_training_set_SFSRM_SMLM(datadir_save, lrImgPath, edgeImgPath, trainsize, SMLMImgPath, scale):

    savedir = datadir_save
    datadir = lrImgPath
    edgedir = edgeImgPath
    cutsize = trainsize
    smlmdir = SMLMImgPath
    bin = scale

    savedir1=os.path.join(savedir,'SMLM_Hessian_patches')
    os.makedirs(savedir1,exist_ok=True)
    get_traindata_SFSRM3(datadir,edgedir,smlmdir,savedir1,bin,cutsize)
    print("===> Setting Creating training Data_for_SFSRM-WSF")
    return DatasetFromFolder_SFSRM3(savedir1)



def get_traindata_SFSRM2(datadir,edgedir,savedir,cutsize=[64,64],timelen=20):
    image = io.imread(datadir)
    edge = io.imread(edgedir)

    n=1
    stepx=int(cutsize[0])
    stepy=int(cutsize[1])
    s = image.shape

    coorx = 0
    corry = 0
    linebreak = 0
    lastline = 0

    image = image / image.max()
    edge = edge / edge.max()
    cutlen=1000

    if len(image.shape)==3:
        for a in (range(cutlen)):
            if corry + cutsize[1] >= s[2]:
                corry = s[2] - 1 - cutsize[1]
                linebreak = 1

            time = (s[0] - timelen) * random.random()
            time = int(round(time))

            image_part = image[time:time + timelen, coorx:coorx + cutsize[0], corry:corry + cutsize[1]]
            io.imsave(os.path.join(savedir, str(n) + '_wf.tif'), image_part.astype('float32'), check_contrast=False)
            edge_part = edge[time:time + timelen, coorx:coorx + cutsize[0], corry:corry + cutsize[1]]
            io.imsave(os.path.join(savedir, str(n) + '_eg.tif'), edge_part.astype('float32'), check_contrast=False)

            if linebreak == 1 and lastline == 1:
                break
            if linebreak == 0:
                corry = corry + stepx
            if linebreak == 1:
                corry = 0
                coorx = coorx + stepy
                linebreak = 0

            if coorx + cutsize[0] >= s[1]:
                coorx = s[1] - 1 - cutsize[0]
                lastline = 1
            if linebreak == 1 and lastline == 1:
                break

            n += 1

    if len(image.shape)==2:
        for a in (range(cutlen)):
            if corry + cutsize[1] >= s[1]:
                corry = s[1] - 1 - cutsize[1]
                linebreak = 1

            image_part = image[coorx:coorx + cutsize[0], corry:corry + cutsize[1]]
            io.imsave(os.path.join(savedir, str(n) + '_wf.tif'), image_part.astype('float32'), check_contrast=False)
            edge_part = edge[coorx:coorx + cutsize[0], corry:corry + cutsize[1]]
            io.imsave(os.path.join(savedir, str(n) + '_eg.tif'), edge_part.astype('float32'), check_contrast=False)

            if linebreak == 1 and lastline == 1:
                break
            if linebreak == 0:
                corry = corry + stepx
            if linebreak == 1:
                corry = 0
                coorx = coorx + stepy
                linebreak = 0

            if coorx + cutsize[0] >= s[0]:
                coorx = s[0] - 1 - cutsize[0]
                lastline = 1
            if linebreak == 1 and lastline == 1:
                break

            n += 1




def get_traindata_SFSRM3(datadir,edgedir,smlmdir,savedir,bin,cutsize=[64,64]):
    image = io.imread(datadir)
    edge = io.imread(edgedir)
    smlm = io.imread(smlmdir)
    n=1
    stepx=int(cutsize[0])
    stepy=int(cutsize[1])
    s = image.shape

    coorx = 0
    corry = 0
    linebreak = 0
    lastline = 0

    image = image / image.max()
    edge = edge / edge.max()
    smlm = smlm / smlm.max()


    cutlen=1000

    for a in (range(cutlen)):
        if corry + cutsize[1] >= s[2]:
            corry = s[2] - 1 - cutsize[1]
            linebreak = 1




        smlm_part = smlm[coorx*bin:coorx*bin + cutsize[0]*bin, corry*bin:corry*bin + cutsize[1]*bin]
        io.imsave(os.path.join(savedir, str(n) + '_smlm.tif'), smlm_part.astype('float32'), check_contrast=False)
        image_part = image[0:, coorx:coorx + cutsize[0], corry:corry + cutsize[1]]
        io.imsave(os.path.join(savedir, str(n) + '_wf.tif'), image_part.astype('float32'), check_contrast=False)
        edge_part = edge[0:, coorx:coorx + cutsize[0], corry:corry + cutsize[1]]
        io.imsave(os.path.join(savedir, str(n) + '_eg.tif'), edge_part.astype('float32'), check_contrast=False)
        n += 1

        if linebreak == 1 and lastline == 1:
            break
        if linebreak == 0:
            corry = corry + stepx
        if linebreak == 1:
            corry = 0
            coorx = coorx + stepy
            linebreak = 0

        if coorx + cutsize[0] >= s[1]:
            coorx = s[1] - 1 - cutsize[0]
            lastline = 1
        if linebreak == 1 and lastline == 1:
            break



