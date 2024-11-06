import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.REG import *
from tqdm import tqdm
import random
from torch.autograd import Variable
import pandas as pd
from utils.Result_show import ReconSR
import os

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


def SRmodel_SSF (train_set,lrImgPath, edgeImgPath, datadir_save, SFSRMmodeldir, DENmodeldir, scale, iteration_times, sigma, ksize, SRlr, continuity_level, sparse_level, consistency_level, joint_Training, predict_size, step, gap):

    datadir = lrImgPath
    edgedir =edgeImgPath
    savedir =datadir_save
    SRmodeldir = SFSRMmodeldir
    DEGmodel_savedir1 =DENmodeldir
    bin =scale
    iteration_times =iteration_times
    sigma =sigma
    ksize =ksize
    SRlr =SRlr
    continuity_level = continuity_level
    sparse_level =sparse_level
    consistency_level =consistency_level
    joint_Training = joint_Training
    continuity_weight = continuity_level * 0.025
    sparse_weight =  sparse_level
    predict_size =predict_size
    step =step
    gap = gap
    modelsavedir=os.path.join(savedir,'Fine-tuning_Models')
    os.makedirs(modelsavedir,exist_ok=True)

    print("===> Setting Fine-tuning process")

    cudnn.benchmark = True
    record = np.zeros(shape=[iteration_times, 5])
    batchsize = 1

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(20)

    model=torch.load(SRmodeldir)
    model_contrast=torch.load(SRmodeldir)




    degmodel1 = torch.load(DEGmodel_savedir1)
    degmodel1 = degmodel1.cuda()


    model = model.cuda()
    model_contrast = model_contrast.cuda()

    model_save = model
    degmodel_save=degmodel1
    optimizer = optim.Adam(model.parameters(), lr=SRlr)

    if joint_Training:
        optimizerdeg1 = optim.Adam(degmodel1.parameters(), lr=SRlr/10)

    for epoch in tqdm(range(iteration_times)):



        fineTuningDir = ReconSR(model_save,bin,predict_size,step,gap,datadir,edgedir,savedir,FilenameSave=str(epoch),imagetypeSave='.tif')


        model.train()

        training_data_loader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True)
        deg_loss_total = 0
        loss_total = 0
        Sparse_total = 0
        Hessian_total = 0
        enhance_total = 0

        for iteration, batch in (enumerate(training_data_loader, 1)):

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

            inputimage2 = torch.zeros(input_image.shape).cuda()
            edgemap2 = torch.zeros(edgemap.shape).cuda()
            input_len=input.shape[0]

            for r in range(input_len):

                inputimage2[r,:,:,:]=torch.rot90(input_image[r,:,:,:],k=2,dims=[1,2])
                edgemap2[r,:,:,:]=torch.rot90(edgemap[r,:,:,:],k=2,dims=[1,2])

            input2=torch.cat((edgemap2,inputimage2),1)




            out = model(input)
            out = torch.abs(out)


            input_1 = input
            model_contrast.eval()
            degmodel1.eval()
            for name, parameter in degmodel1.named_parameters():
                parameter.requires_grad = False

            for name, parameter in model_contrast.named_parameters():
                parameter.requires_grad = False


            out_contrast=model_contrast(input2)
            out_contrast=torch.abs(out_contrast)
            out_contrast2=torch.zeros(out_contrast.shape).cuda()
            for r in range(len(out_contrast2)):
                out_contrast2[r, :, :, :] = torch.rot90(out_contrast[r, :, :, :], k=2, dims=[1, 2])


            input_1_denoising = torch.abs((degmodel1(input_1[:, 1, :, :].unsqueeze(1))))  ####降噪功能
            out1_3 = nn.AvgPool2d(bin)(out)
            degout_i_3 = GaussianBlur(out1_3, ksize=ksize, sigma=sigma)
            intensity_adjust=torch.mean(input_1_denoising)/torch.mean(degout_i_3)
            degout_i_3=degout_i_3*intensity_adjust
            degloss_1 = nn.L1Loss()(degout_i_3, input_1_denoising)


            Sparseloss =torch.abs(torch.mean(out)-torch.mean(out_contrast)*(1/sparse_weight))
            enhanceloss = nn.L1Loss()(out,out_contrast2)

            out_squeeze=out.squeeze()
            if len(list(out_squeeze.shape))==3:
                Hessianloss = HessianReg()(out_squeeze.unsqueeze(0).unsqueeze(0))
                total_loss = degloss_1 + Hessianloss * continuity_weight + Sparseloss + enhanceloss * consistency_level
            if len(list(out_squeeze.shape))==2:
                total_loss = degloss_1 + Sparseloss + enhanceloss * consistency_level



            deg_loss_total += degloss_1.item()
            loss_total += total_loss.item()
            Sparse_total += Sparseloss.item()
            enhance_total += enhanceloss * consistency_level
            if len(list(out_squeeze.shape))==3:
                Hessian_total += Hessianloss.item() * continuity_weight

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)

            optimizer.step()
            optimizer.zero_grad()


            if joint_Training:

                degmodel1.train()
                for name, parameter in degmodel1.named_parameters():
                    parameter.requires_grad = True

                denosing_input1 = degmodel1(input_1[:, 1, :, :].unsqueeze(1))
                denosing_input1 = torch.abs(denosing_input1)
                degout_i_1 = GaussianBlur(nn.AvgPool2d(bin)(out).detach(), ksize=ksize, sigma=sigma)
                degloss1 = nn.L1Loss()(degout_i_1, denosing_input1) + nn.L1Loss()(denosing_input1,input_1[:, 1, :, :].unsqueeze(1))
                degloss1.backward()
                optimizerdeg1.step()
                optimizerdeg1.zero_grad()

        model_save = model
        torch.save(model_save, os.path.join(modelsavedir,str(epoch+1)+'.pth'))
        degmodel_save = degmodel1
        record[epoch, 0] = deg_loss_total
        record[epoch, 1] = Hessian_total
        record[epoch, 2] = Sparse_total
        record[epoch, 3] = enhance_total
        record[epoch, 4] = loss_total

    dataframe = pd.DataFrame({'DRSE': record[:, 0], 'Hessian': record[:, 1],'Sparse': record[:, 2],
                              'enhance': record[:, 3],'total': record[:, 4]})
    dataframe.to_csv(savedir + r'\\SFSRM-SSF_Finetuning_Record.csv', index=False, sep=',')

    return model_save, os.path.join(datadir_save, fineTuningDir)