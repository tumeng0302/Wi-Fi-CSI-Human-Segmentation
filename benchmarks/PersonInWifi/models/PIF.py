import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.conv_block(256, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv_last(dec1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)
        self.residual_block1 = ResidualBlock(51)
        self.residual_block2 = ResidualBlock(51)
        self.unet = UNet(51, 51)
        self.conv1 = nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=51, out_channels=51, kernel_size=3, stride=(1, 2))
        self.upsample = nn.Upsample(size=(192, 256), mode='bilinear', align_corners=True)

        self.sm = nn.Conv2d(in_channels=51, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.jhm = nn.Conv2d(in_channels=51, out_channels=25, kernel_size=1, stride=1, padding=0)
        self.paf = nn.Conv2d(in_channels=51, out_channels=52, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resample(x)
        skip = x
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = x + skip
        x = self.unet(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)

        sm = self.sigmoid(self.sm(x))
        jhm = self.sigmoid(self.jhm(x))
        paf = self.sigmoid(self.paf(x))

        return sm, jhm, paf
    
if __name__ == '__main__':
    model = Model()
    input = torch.randn([2, 51, 6, 2025])
    output = model(input)
    print(output[0].shape, output[1].shape, output[2].shape)

# model = UNet(51, 51)
# input = torch.randn([2, 51, 128, 128])
# output = model(input)
# print(output.shape)