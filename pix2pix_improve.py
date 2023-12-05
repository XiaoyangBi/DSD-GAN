import datetime
import os
import sys
import time

import cv2
import numpy as np
import torch

import torchvision.transforms as transforms
from torch.autograd import Variable

from Frames_dataset import FramesDataset
from models import *
from opts import parse_opts

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# torch.cuda.empty_cache()
# torch.backends.cudnn.enabled = False
opt = parse_opts()
print(opt)

os.makedirs("D:/FYP/dataset/images_generate/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("D:/FYP/dataset/saved_models/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("D:/FYP/dataset/result/%s" % opt.dataset_name, exist_ok=True)

# 损失函数
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
vgg_loss = VGGLoss()
lambda_pixel = 100
lambda_vgg = 10
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# 初始化生成器和判别器
generator = GeneratorWithEnhancer()
discriminator1 = Discriminator1(in_channels=3)
discriminator2 = Discriminator2(in_channels=3)

cuda = True if torch.cuda.is_available() else False
# cuda = False
print(cuda)

# 使用gpu
if cuda:
    generator = generator.cuda()
    discriminator1 = discriminator1.cuda()
    discriminator2 = discriminator2.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # 导入训练好的模型
    generator.load_state_dict(torch.load("D:/FYP/dataset/saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator1.load_state_dict(torch.load("D:/FYP/dataset/saved_models/%s/discriminator1_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator2.load_state_dict(torch.load("D:/FYP/dataset/saved_models/%s/discriminator2_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # 初始化权重
    generator.apply(weights_init_normal)
    discriminator1.apply(weights_init_normal)
    discriminator2.apply(weights_init_normal)



# 优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D1 = torch.optim.Adam(discriminator1.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D2 = torch.optim.Adam(discriminator2.parameters(), lr=0.0002, betas=(0.5, 0.999))


# 图像变换
transform=transforms.Compose([
                               transforms.Resize(opt.img_size),   #把图像的短边统一为image_size，另外一边做同样倍速缩放，不一定为image_size
                               transforms.ToTensor(),
                           ])



# 创建数据迭代器
dataset = FramesDataset(opt,dataset='BXY',transform=transform)

dataloader = torch.utils.data.DataLoader(dataset,batch_size=opt.batch_size,shuffle=True)
val_dataloader = torch.utils.data.DataLoader(dataset,batch_size=opt.batch_size,shuffle=False)


# Tensor 类型
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs[1].type(Tensor))
    real_B = Variable(imgs[0].type(Tensor))
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    cv2.imwrite("D:/FYP/dataset/images_generate/%s/%s.png" % (opt.dataset_name, batches_done),255*img_sample[0].squeeze(0).cpu().swapaxes(0,2).swapaxes(0,1).numpy())


# ----------
#  Training
# ----------
def calculate_metrics(real_B, fake_B):
    real_B = real_B.squeeze(0).detach().cpu().numpy()  # 将张量转换为NumPy数组
    fake_B = fake_B.squeeze(0).detach().cpu().numpy()

    # 计算SSIM
    ssim_score = ssim(real_B, fake_B, channel_axis=True, win_size = 3)

    # 计算MSE
    mse_score = mean_squared_error(real_B, fake_B)

    # 计算PSNR
    max_pixel_value = 255.0  # 如果图像是8位的
    psnr_score = 10 * np.log10((max_pixel_value ** 2) / mse_score)

    return ssim_score, mse_score, psnr_score

def calculate_PI(real_B, fake_B, feature_extractor):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # print("Original real_B dimensions:", real_B.size())
    # print("Original fake_B dimensions:", fake_B.size())
    # 确保输入是单个图像（移除批次维度，如果存在）
    real_B = real_B.squeeze(0)
    fake_B = fake_B.squeeze(0)
    # print("Squeezed real_B dimensions:", real_B.size())
    # print("Squeezed fake_B dimensions:", fake_B.size())
    real_B_processed = preprocess(real_B)
    fake_B_processed = preprocess(fake_B)
    # print("processed real_B dimensions:", real_B_processed.size())
    # print("processed fake_B dimensions:", fake_B_processed.size())
    # 检查输入是否为3维
    if real_B.dim() != 3 or fake_B.dim() != 3:
        raise ValueError("Input images must be 3D tensors")

    # 应用预处理
    real_B_processed = preprocess(real_B)
    fake_B_processed = preprocess(fake_B)

    # 提取特征并计算 PI
    real_features = feature_extractor(real_B_processed.unsqueeze(0))
    fake_features = feature_extractor(fake_B_processed.unsqueeze(0))

    # 计算欧氏距离作为 PI
    pi_score = torch.nn.functional.mse_loss(real_features, fake_features)

    return pi_score.item()

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # if epoch % 2 == 0:
        #     # Perform testing every 50 epochs
        #     generator.eval()  # Set the generator to evaluation mode
        #     ssim_sum, mse_sum, psnr_sum, pi_sum = 0, 0, 0, 0
        #     # 确保特征提取器处于评估模式
        #     feature_extractor = VGGFeatureExtractor().eval()
        #     with torch.no_grad():
        #         for val_batch in val_dataloader:
        #             real_A_val = Variable(val_batch[1].type(Tensor))
        #             real_B_val = Variable(val_batch[0].type(Tensor))
        #             fake_B_val = generator(real_A_val)
        #
        #             # Calculate metrics
        #             ssim_val, mse_val, psnr_val = calculate_metrics(real_B_val, fake_B_val)
        #             ssim_sum += ssim_val
        #             mse_sum += mse_val
        #             psnr_sum += psnr_val
        #             for i in range(real_B_val.size(0)):
        #                 single_real_B = real_B_val[i].unsqueeze(0)  # 为单个图像添加批次维度
        #                 single_fake_B = fake_B_val[i].unsqueeze(0)  # 为单个图像添加批次维度
        #                 pi_val = calculate_PI(single_real_B, single_fake_B, feature_extractor)
        #                 pi_sum += pi_val
        #     num_batches = len(val_dataloader)
        #     ssim_avg = ssim_sum / num_batches
        #     mse_avg = mse_sum / num_batches
        #     psnr_avg = psnr_sum / num_batches
        #     pi_avg = pi_sum / num_batches
        #
        #     # Print test results
        #     print("\nValidation results at Epoch %d, Batch %d:" % (epoch, i))
        #     print("SSIM: %.4f, MSE: %.4f, PSNR: %.4f PI: %.4f" % (ssim_avg, mse_avg, psnr_avg, pi_avg))
        #
        #     generator.train()  # Set the generator back to training mode
        # 输入 tensor shape[512,512]
        real_A = Variable(batch[1].type(Tensor))
        real_B = Variable(batch[0].type(Tensor))

        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        fake_B = generator(real_A)
        downsample = nn.AvgPool2d(2)

        pred_fake_G1 = discriminator1(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake_G1, valid)
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        # loss_vgg = vgg_loss(fake_B, real_B)
        ssim_score, mse_score, psnr_score = calculate_metrics(real_B, fake_B)
        # 总损失
        loss_G_GAN_Feat = 0
        loss_G = loss_GAN + lambda_pixel * loss_pixel
        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        real_A_downsampled = downsample(real_A)
        real_B_downsampled = downsample(real_B)
        fake_B_downsampled = downsample(fake_B)
        # Real loss
        pred_real_D1 = discriminator1(real_B, real_A)
        pred_fake_D1 = discriminator1(fake_B.detach(), real_A)
        loss_real_D1 = criterion_GAN(pred_real_D1, valid)
        loss_fake_D1 = criterion_GAN(pred_fake_D1, fake)
        loss_D1 = 0.5 * (loss_real_D1 + loss_fake_D1)

        # loss_D = loss_D1
        # Fake loss
        pred_real_D2 = discriminator2(real_B_downsampled, real_A_downsampled)
        pred_fake_D2 = discriminator2(fake_B_downsampled.detach(), real_A_downsampled)
        valid_D2 = Variable(Tensor(np.ones((pred_real_D2.size(0), *pred_real_D2.size()[1:]))), requires_grad=False)
        fake_D2 = Variable(Tensor(np.zeros((pred_fake_D2.size(0), *pred_fake_D2.size()[1:]))), requires_grad=False)
        loss_real_D2 = criterion_GAN(pred_real_D2, valid_D2)
        loss_fake_D2 = criterion_GAN(pred_fake_D2, fake_D2)
        loss_D2 = 0.5 * (loss_real_D2 + loss_fake_D2)
        weight_D1 = 0.7  # 调整权重以平衡 D1 和 D2 的贡献
        weight_D2 = 0.3
        loss_D = weight_D1 * loss_D1 + weight_D2 * loss_D2

        # loss_G_GAN_Feat = 0
        # pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
        # if not self.opt.no_ganFeat_loss:
        #     feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        #     D_weights = 1.0 / self.opt.num_D
        #     for i in range(self.opt.num_D):
        #         for j in range(len(pred_fake[i]) - 1):
        #             loss_G_GAN_Feat += D_weights * feat_weights * \
        #                                self.criterionFeat(pred_fake[i][j],
        #                                                   pred_real[i][j].detach()) * self.opt.lambda_feat
        # loss_G_VGG = 0
        # if not self.opt.no_vgg_loss:
        #     loss_G_VGG = self.criterionVGG(enhance, real_image) * self.opt.lambda_feat
        loss_D.backward()

        optimizer_D1.step()
        optimizer_D2.step()

        # --------------
        #  Log Progress
        # --------------

        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # # 打印log [ps: 这段代码很神奇！！]
        # sys.stdout.write(
        #     "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f]  [SSIM_score: %f, MSE: %f, PSNR: %f] ETA: %s"
        #     % (
        #         epoch,
        #         opt.n_epochs,
        #         i,
        #         len(dataloader),
        #         loss_D.item(),
        #         loss_G.item(),
        #         loss_pixel.item(),
        #         loss_GAN.item(),
        #         ssim_score,
        #         mse_score,
        #         psnr_score,
        #         time_left,
        #     )
        # )

        # 打印log [ps: 这段代码很神奇！！]
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        # 如果到达一定时间就保存图片
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # 保存模型参数
        torch.save(generator.state_dict(), "D:/FYP/dataset/saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator1.state_dict(), "D:/FYP/dataset/saved_models/%s/discriminator1_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator2.state_dict(), "D:/FYP/dataset/saved_models/%s/discriminator2_%d.pth" % (opt.dataset_name, epoch))
