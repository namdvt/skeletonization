import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(UpConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DoubleConv2d, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionGroup(nn.Module):
    def __init__(self, num_channels):
        super(AttentionGroup, self).__init__()
        self.conv1 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv_1x1 = nn.Conv2d(num_channels, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        s = torch.softmax(self.conv_1x1(x), dim=1)

        att = s[:,0,:,:].unsqueeze(1) * x1 + s[:,1,:,:].unsqueeze(1) * x2 \
            + s[:,2,:,:].unsqueeze(1) * x3

        return x + att


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = DoubleConv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = DoubleConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = DoubleConv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = DoubleConv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = DoubleConv2d(512, 1024, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.att1 = AttentionGroup(64)
        self.att2 = AttentionGroup(128)
        self.att3 = AttentionGroup(256)
        self.att4 = AttentionGroup(512)
        self.att5 = AttentionGroup(1024)


    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.att1(out1)

        out2 = self.conv2(self.pooling(out1))
        out2 = self.att2(out2)

        out3 = self.conv3(self.pooling(out2))
        out3 = self.att3(out3)

        out4 = self.conv4(self.pooling(out3))
        out4 = self.att4(out4)

        out5 = self.conv5(self.pooling(out4))
        out5 = self.att5(out5)

        return out1, out2, out3, out4, out5


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upconv1 = UpConv2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = UpConv2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = UpConv2d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = UpConv2d(128, 64, kernel_size=2, stride=2)

        self.conv1 = DoubleConv2d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = DoubleConv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = DoubleConv2d(256, 128, kernel_size=3, padding=1)
        self.conv4 = DoubleConv2d(128, 64, kernel_size=3, padding=1)

        self.conv1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_conv_128 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_conv_64 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_conv_32 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.ca1 = ChannelAttention(512)
        self.sa1 = SpatialAttention()

        self.ca2 = ChannelAttention(256)
        self.sa2 = SpatialAttention()

        self.ca3 = ChannelAttention(128)
        self.sa3 = SpatialAttention()

        self.ca4 = ChannelAttention(64)
        self.sa4 = SpatialAttention()


    def forward(self, out1, out2, out3, out4, x):
        x = self.upconv1(x)
        x = torch.cat([x, out4], dim=1)
        x = self.conv1(x)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        aux_32 = self.aux_conv_32(x)

        x = self.upconv2(x)
        x = torch.cat([x, out3], dim=1)
        x = self.conv2(x)
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        aux_64 = self.aux_conv_64(x)

        x = self.upconv3(x)
        x = torch.cat([x, out2], dim=1)
        x = self.conv3(x)
        x = self.ca3(x) * x
        x = self.sa3(x) * x
        aux_128 = self.aux_conv_128(x)

        x = self.upconv4(x)
        x = torch.cat([x, out1], dim=1)
        x = self.conv4(x)
        x = self.ca4(x) * x
        x = self.sa4(x) * x
        x = self.conv1x1(x)

        return x, aux_128, aux_64, aux_32


class UnetAttention(nn.Module):
    def __init__(self):
        super(UnetAttention, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        out1, out2, out3, out4, x = self.encoder(x.float())
        x, aux_128, aux_64, aux_32 = self.decoder(out1, out2, out3, out4, x)

        return x.squeeze(), aux_128.squeeze(), aux_64.squeeze(), aux_32.squeeze()
