import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.REG import *
from tqdm import tqdm
import random
from torch.autograd import Variable
import pandas as pd
import os
from utils.Result_show import ReconSR
import torch





def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) 
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()
    return kernel


def GaussianBlur(batch_img, ksize, sigma=None):
    kernel = getGaussianKernel(ksize, sigma)
    B, C, H, W = batch_img.shape 
    kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)
    pad = (ksize - 1) // 2 
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='constant', value=0)
    weighted_pix = F.conv2d(batch_img_pad.cuda(), weight=kernel.cuda(), bias=None,
                            stride=1, padding=0, groups=C)
    return weighted_pix


def SRmodel_WSF(train_set,lrImgPath, edgeImgPath, datadir_save, scale, iteration_times_WSF, SRlr_WSF, timelen_WSF, SFSRMmodeldir, predict_size, step, gap):
    datadir = lrImgPath
    edgedir = edgeImgPath
    savedir = datadir_save
    modelsavedir=os.path.join(savedir,'Fine-tuning_Models')
    os.makedirs(modelsavedir,exist_ok=True)

    bin = scale
    iteration_times = iteration_times_WSF
    SRlr = SRlr_WSF
    timelen = timelen_WSF
    SRmodeldir = SFSRMmodeldir
    predict_size = predict_size
    step = step
    gap = gap

    cudnn.benchmark = True
    record = np.zeros(shape=[iteration_times, 3])
    batchsize = 1


    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(20)
    model=torch.load(SRmodeldir)
    model = model.cuda()


    model_save = model

    optimizer = optim.Adam(model.parameters(), lr=SRlr)

    print("===> Setting Fine-tuning process")
    for epoch in tqdm(range(iteration_times)):

        model.train()

        fineTuningDir = ReconSR(model_save,bin,predict_size,step,gap,datadir,edgedir,savedir,FilenameSave=str(epoch),imagetypeSave='.tif')

        training_data_loader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True)
        SMLM_total = 0


        for iteration, batch in (enumerate(training_data_loader, 1)):


            target= Variable(batch[2].float(), requires_grad=False)

            input_image = Variable(batch[0].float(), requires_grad=False)
            input_image = input_image.cuda()
            edgemap = Variable(batch[1].float(), requires_grad=False)
            edgemap = edgemap.cuda()

            if input_image.shape[1]>1:
                input_image = input_image.squeeze()
                input_image = input_image.unsqueeze(1)
                edgemap = edgemap.squeeze()
                edgemap = edgemap.unsqueeze(1)

            input=torch.cat((edgemap,input_image),1)
            input = input.cuda()
            input_len=input.shape[0]


            out_result=torch.zeros([input_len,target.shape[1],target.shape[2],target.shape[3]]).cuda()

            # target=GaussianBlur(target, ksize=5, sigma=1)
            target=target.cuda()
            target=target.squeeze()
            target=target.float()
            target=target/target.max()

            for n in range(input_len):
                input_frame=input[n,:,:,:].unsqueeze(0)
                out = model(input_frame)
                out = torch.abs(out)
                out_result[n,:,:,:] = out.detach()

            repeat_time=int(round(input_len/timelen))

            if repeat_time>10:
                repeat_time=10

            for p in range(repeat_time):

                out_avg = torch.zeros(target.shape).cuda()
                imagelist = range(0, input_len - 1)
                timelist = random.sample(imagelist, timelen)

                for i in range(input_len):

                    if i in timelist:

                        input_frame = input[i, :, :, :].unsqueeze(0)
                        out = model(input_frame)
                        out = torch.abs(out)
                        out_avg = out + out_avg


                    if i not in timelist:

                        out_avg = out_result[i,:,:,:].unsqueeze(0) + out_avg



                out_avg = out_avg.squeeze()
                out_avg = out_avg / out_avg.max()
                smlmloss = nn.L1Loss()(target, out_avg)/repeat_time
                smlmloss.backward()

                SMLM_total += smlmloss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            optimizer.zero_grad()


        model_save = model
        torch.save(model_save, os.path.join(modelsavedir,str(epoch+1)+'.pth'))
        record[epoch,0] = SMLM_total


    dataframe = pd.DataFrame({'SMLM': record[:, 0]})

    dataframe.to_csv(savedir + r'\\SFSRM-WSF_Finetuning_Record.csv', index=False, sep=',')

    return model_save, os.path.join(datadir_save, fineTuningDir)