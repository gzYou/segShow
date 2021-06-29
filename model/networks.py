import torch as t
import torch.nn as nn
from torch.nn import functional as F

def conv3x3(in_planes, out_planes, stride=1,padding=1,dilation=1):
    return nn.Conv2d(in_planes,out_planes,3,stride=stride,padding=padding,dilation=dilation,bias=False)
def conv1x1(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,1,stride=stride,bias=False)


class BasiBlock(nn.Module):
    expension = 1
    def __init__(self,inplanes,planes,stride=1,dilation=1,downsample=None):
        super(BasiBlock,self).__init__()
        self.conv1 = conv3x3(inplanes,planes,stride=stride,padding=dilation,dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,planes,stride=1,padding=1,dilation=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self,x):
        ind = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        result = x + ( ind if self.downsample is None else self.downsample(ind))
        return F.relu(result,inplace=True)
class Bottleneck(nn.Module):
    expension = 4
    def __init__(self, inplanes, planes, stride=1,dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride,dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expension)
        self.bn3 = nn.BatchNorm2d(planes * self.expension)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class ResNet(nn.Module):
    def __init__(self,block,layers,n_classes=1000,stride = 32):
        super(ResNet,self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3,64,7,2,3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace= True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        if stride == 8 :
            self.layer3 = self._make_layer(block,256,layers[2],stride=1,dilation=2)
            self.layer4 = self._make_layer(block,512,layers[3],stride=1,dilation=4,grid=[1,2])
        elif stride == 16:
            self.layer3 = self._make_layer(block,256,layers[2],stride=2)
            self.layer4 = self._make_layer(block,512,layers[3],stride=1,dilation=2,grid=[1,2])
        else :
            self.layer3 = self._make_layer(block,256,layers[2],stride=2)
            self.layer4 = self._make_layer(block,512,layers[3],stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expension, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,block,planes,blocks,stride=1,dilation=1,grid=None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expension:
            downsample = nn.Sequential(
                conv1x1(self.in_planes,planes * block.expension,stride=stride),
                nn.BatchNorm2d(planes * block.expension)
            ) 
        if grid is None:
            grid = [1] * blocks
        assert len(grid) == blocks
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.in_planes,planes,stride=stride,dilation=1,downsample=downsample))
        elif dilation == 4:
            layers.append(block(self.in_planes,planes,stride=stride,dilation=2,downsample=downsample))
        self.in_planes = planes * block.expension
        for i in range(1,blocks):
            layers.append(block(self.in_planes,planes,stride=1,dilation=dilation * grid[i]))
        return nn.Sequential(*layers)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
    
        return x

def resnet(stride):
    layers = [2,2,2,2]
    # layers = [3, 4, 6, 3]
    net = ResNet(BasiBlock,layers,stride=stride)
    # net.load_state_dict(t.load('./pretrained/resnet18-5c106cde.pth')) # 模型名字 还没下载完成
    return net
class convblock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(convblock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        return self.conv(x)
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            conv3x3(ch_in,ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        return self.up(x)
class SEBlock(nn.Module):
    def __init__(self,ch_in,r=8):
        super(SEBlock,self).__init__()
        self.averagePool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(ch_in,ch_in//r),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in//r,ch_in),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        b,c,_,_ = x.size()
        y = self.averagePool(x).view(b,c)
        y = self.linear(y).view(b,c,1,1)
        return x * y.expand_as(x)

class UNetComposedLossSupervised(nn.Module):
    def __init__(self,n_classes=1,stride=32,r=32):
        super(UNetComposedLossSupervised,self).__init__()
        backbone = resnet(stride=stride)
        self.conv1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.down_layer1 = backbone.layer1
        self.down_layer2 = backbone.layer2
        self.down_layer3 = backbone.layer3
        self.down_layer4 = backbone.layer4

        self.logix_pixel5 = nn.Sequential(
            conv3x3(512,128),
            conv3x3(128,32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv3x3(32,n_classes),
        )
        self.logix_pixel4 = nn.Sequential(
            conv3x3(256,64),
            conv3x3(64,16),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            conv3x3(16,n_classes)
        )
        self.logix_pixel3 = nn.Sequential(
            conv3x3(128,32),
            conv3x3(32,8),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            conv3x3(8,n_classes)
        )
        self.logix_pixel2 = nn.Sequential(
            conv3x3(64,16),
            conv3x3(16,4),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            conv3x3(4,n_classes)
        )
        self.fuse = conv3x3(4,1)

        self.se_block = SEBlock(ch_in=512,r=r)
        
        self.up4 = up_conv(512,256)
        self.up4_block = convblock(512,256)
        
        self.up3 = up_conv(256,128)
        self.up3_block = convblock(256,128)

        self.up2 = up_conv(128,64)
        self.up2_block = convblock(128,64)
        
        self.up1 = conv1x1(64,n_classes)
    
    def forward(self,x):
        x = self.conv1(x)
        x1 = self.down_layer1(x)
        x2 = self.down_layer2(x1)
        x3 = self.down_layer3(x2)
        x4 = self.down_layer4(x3)
        
        # 通道 attention
        x4 = self.se_block(x4)

        o4 = self.logix_pixel5(x4)
        o4 = F.interpolate(o4,scale_factor=32,mode='bilinear',align_corners=True)
        d4 = self.up4(x4)
        d4 = t.cat((x3,d4),dim=1)
        d4 = self.up4_block(d4)
        
        o3 = self.logix_pixel4(d4)
        o3 = F.interpolate(o3,scale_factor=16,mode='bilinear',align_corners=True)
        d3 = self.up3(d4)
        d3 = t.cat((x2,d3),dim=1)
        d3 = self.up3_block(d3)

        o2 = self.logix_pixel3(d3)
        o2 = F.interpolate(o2,scale_factor=8,mode='bilinear',align_corners=True)

        d2 = self.up2(d3)
        d2 = t.cat((x1,d2),dim=1)
        d2 = self.up2_block(d2)

        o1 = self.logix_pixel2(d2)
        o1 = F.interpolate(o1,scale_factor=4,mode='bilinear',align_corners=True)

        d1 = self.up1(d2)
        out = F.interpolate(d1,scale_factor=4,mode='bilinear',align_corners=True)
        out = F.sigmoid(out)

        return out,o1,o2,o3,o4
