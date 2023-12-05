import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 添加一个卷积层
        self.conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv(self.avg_pool(x))
        max_out = self.conv(self.max_pool(x))

        out = self.relu(avg_out + max_out)
        out = self.conv2(out)
        return self.sigmoid(out).expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x).expand_as(x)

class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, use_attention=False):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

        if use_attention:
            self.attention = CBAMBlock(out_size)

    def forward(self, x):
        x = self.model(x)
        if hasattr(self, 'attention'):
            x = self.attention(x)
        return x

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, use_attention=False):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

        if use_attention:
            self.attention = CBAMBlock(out_size)

    def forward(self, x, skip_input):
        x = self.model(x)
        if hasattr(self, 'attention'):
            x = self.attention(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        # 下采样部分
        self.down1 = UNetDown(in_channels, 64, normalize=False, use_attention=False)
        self.down2 = UNetDown(64, 128, use_attention=False)
        self.down3 = UNetDown(128, 256, use_attention=False)
        self.down4 = UNetDown(256, 512, dropout=0.5, use_attention=False)
        self.down5 = UNetDown(512, 512, dropout=0.5, use_attention=False)
        self.down6 = UNetDown(512, 512, dropout=0.5, use_attention=False)
        self.down7 = UNetDown(512, 512, dropout=0.5, use_attention=False)
        self.down8 = UNetDown(512, 512, normalize=True, dropout=0.5, use_attention=False)

        # 上采样部分
        self.up1 = UNetUp(512, 512, dropout=0.5, use_attention=False)
        self.up2 = UNetUp(1024, 512, dropout=0.5, use_attention=False)
        self.up3 = UNetUp(1024, 512, dropout=0.5, use_attention=False)
        self.up4 = UNetUp(1024, 512, dropout=0.5, use_attention=False)
        self.up5 = UNetUp(1024, 256, use_attention=False)
        self.up6 = UNetUp(512, 128, use_attention=False)
        self.up7 = UNetUp(256, 64, use_attention=False)

        # 最终层
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net生成器的前向传播
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################
class Discriminator1(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator1, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class Discriminator2(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator2, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 简化的结构可能包含较少的层和/或较少的通道
        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

class Enhancer(nn.Module):
    def __init__(self):
        super(Enhancer, self).__init__()

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.tanh=nn.Tanh()
        self.refine1= nn.Conv2d(3, 20, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        self.upsample = F.upsample_nearest
        self.batch1 = nn.InstanceNorm2d(100, affine=True)
    def forward(self, x):
        enhance = self.relu((self.refine1(x)))
        enhance = self.relu((self.refine2(enhance)))
        shape_out = enhance.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(enhance, 32)
        x102 = F.avg_pool2d(enhance, 16)
        x103 = F.avg_pool2d(enhance, 8)
        x104 = F.avg_pool2d(enhance, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)
        enhance = torch.cat((x1010, x1020, x1030, x1040, enhance), 1)
        enhance = self.tanh(self.refine3(enhance))
        return enhance

class GeneratorWithEnhancer(nn.Module):
    def __init__(self):
        super(GeneratorWithEnhancer, self).__init__()

        # Create instances of GeneratorUNet and Enhancer
        self.generator = GeneratorUNet()
        self.enhancer = Enhancer()

    def forward(self, x):
        # Generate an image using the GeneratorUNet
        generated_image = self.generator(x)

        # Enhance the generated image using the Enhancer
        enhanced_image = self.enhancer(generated_image)

        return enhanced_image

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        self.features = nn.Sequential(*list(vgg_features.children())[:35])  # 取前35层

    def forward(self, img):
        return self.features(img)
