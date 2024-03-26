import numpy as np
import torch
import torch.nn as nn

from nets.backbone import Backbone, Conv, SiLU
from nets.common import SPPELAN, RepNCSPELAN4

   
class DualDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 2  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((ch[0] // 4, self.reg_max * 4, 16)), max((ch[0], min((self.nc * 2, 128))))  # channels
        c4, c5 = max((ch[self.nl] // 4, self.reg_max * 4, 16)), max((ch[self.nl], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch[:self.nl])
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, 4 * self.reg_max, 1)) for x in ch[self.nl:])
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1)) for x in ch[self.nl:])
        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        d1 = []
        d2 = []
        for i in range(self.nl):
            d1.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            d2.append(torch.cat((self.cv4[i](x[self.nl+i]), self.cv5[i](x[self.nl+i])), 1))
        if self.training:
            return [d1, d2]
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (d1.transpose(0, 1) for d1 in make_anchors(d1, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2], 2).split((self.reg_max * 4, self.nc), 1)
        dbox2 = dist2bbox(self.dfl2(box2), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = [torch.cat((dbox, cls.sigmoid()), 1), torch.cat((dbox2, cls2.sigmoid()), 1)]
        return y if self.export else (y, [d1, d2])

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        #-----------------------------------------------#
        #   定义了不同yolov9版本的参数
        #-----------------------------------------------#
        transition_channels = {'l' : 32, 'x' : 40}[phi]
        block_channels      = 32
        panet_channels      = {'l' : 32, 'x' : 64}[phi]
        e       = {'l' : 2, 'x' : 1}[phi]
        n       = {'l' : 4, 'x' : 6}[phi]
        ids     = {'l' : [-1, -2, -3, -4, -5, -6], 'x' : [-1, -3, -5, -7, -8]}[phi]
        # conv    = {'l' : RepConv, 'x' : Conv}[phi]

        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #
        #
        #------------------------------------------- ---#   
        #   生成主干模型
        #   获得5个有效特征层，他们的shape分别是：
        #   80, 80, 512
        #   40, 40, 1024
        #   20, 20, 1024
        #---------------------------------------------------#
        self.backbone     = Backbone(transition_channels, block_channels, n, phi, pretrained=pretrained)
        self.sspelan      = SPPELAN()
        self.upsample     = nn.Upsample()
        self.repncspelan4 = RepNCSPELAN4()

        # #------------------------加强特征提取网络------------------------# 
        # self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        # # 20, 20, 1024 => 20, 20, 512
        # self.sppcspc                = SPPCSPC(transition_channels * 32, transition_channels * 16)
        # # 20, 20, 512 => 20, 20, 256 => 40, 40, 256
        # self.conv_for_P5            = Conv(transition_channels * 16, transition_channels * 8)
        # # 40, 40, 1024 => 40, 40, 256
        # self.conv_for_feat2         = Conv(transition_channels * 32, transition_channels * 8)
        # # 40, 40, 512 => 40, 40, 256
        # self.conv3_for_upsample1    = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        # # 40, 40, 256 => 40, 40, 128 => 80, 80, 128
        # self.conv_for_P4            = Conv(transition_channels * 8, transition_channels * 4)
        # # 80, 80, 512 => 80, 80, 128
        # self.conv_for_feat1         = Conv(transition_channels * 16, transition_channels * 4)
        # # 80, 80, 256 => 80, 80, 128
        # self.conv3_for_upsample2    = Multi_Concat_Block(transition_channels * 8, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids)

        # # 80, 80, 128 => 40, 40, 256
        # self.down_sample1           = Transition_Block(transition_channels * 4, transition_channels * 4)
        # # 40, 40, 512 => 40, 40, 256
        # self.conv3_for_downsample1  = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        # # 40, 40, 256 => 20, 20, 512
        # self.down_sample2           = Transition_Block(transition_channels * 8, transition_channels * 8)
        # # 20, 20, 1024 => 20, 20, 512
        # self.conv3_for_downsample2  = Multi_Concat_Block(transition_channels * 32, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids)
        # #------------------------加强特征提取网络------------------------# 

        # # 80, 80, 128 => 80, 80, 256
        # self.rep_conv_1 = conv(transition_channels * 4, transition_channels * 8, 3, 1)
        # # 40, 40, 256 => 40, 40, 512
        # self.rep_conv_2 = conv(transition_channels * 8, transition_channels * 16, 3, 1)
        # # 20, 20, 512 => 20, 20, 1024
        # self.rep_conv_3 = conv(transition_channels * 16, transition_channels * 32, 3, 1)

        # # 4 + 1 + num_classes
        # # 80, 80, 256 => 80, 80, 3 * 25 (4 + 1 + 20) & 85 (4 + 1 + 80)
        # self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
        # # 40, 40, 512 => 40, 40, 3 * 25 & 85
        # self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        # # 20, 20, 512 => 20, 20, 3 * 25 & 85
        # self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self
    
    def forward(self, x):
        #  backbone
        # feat1, feat2, feat3 = self.backbone.forward(x)
        
        # #------------------------加强特征提取网络------------------------# 
        # # 20, 20, 1024 => 20, 20, 512
        # P5          = self.sppcspc(feat3)
        # # 20, 20, 512 => 20, 20, 256
        # P5_conv     = self.conv_for_P5(P5)
        # # 20, 20, 256 => 40, 40, 256
        # P5_upsample = self.upsample(P5_conv)
        # # 40, 40, 256 cat 40, 40, 256 => 40, 40, 512
        # P4          = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        # # 40, 40, 512 => 40, 40, 256
        # P4          = self.conv3_for_upsample1(P4)

        # # 40, 40, 256 => 40, 40, 128
        # P4_conv     = self.conv_for_P4(P4)
        # # 40, 40, 128 => 80, 80, 128
        # P4_upsample = self.upsample(P4_conv)
        # # 80, 80, 128 cat 80, 80, 128 => 80, 80, 256
        # P3          = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        # # 80, 80, 256 => 80, 80, 128
        # P3          = self.conv3_for_upsample2(P3)

        # # 80, 80, 128 => 40, 40, 256
        # P3_downsample = self.down_sample1(P3)
        # # 40, 40, 256 cat 40, 40, 256 => 40, 40, 512
        # P4 = torch.cat([P3_downsample, P4], 1)
        # # 40, 40, 512 => 40, 40, 256
        # P4 = self.conv3_for_downsample1(P4)

        # # 40, 40, 256 => 20, 20, 512
        # P4_downsample = self.down_sample2(P4)
        # # 20, 20, 512 cat 20, 20, 512 => 20, 20, 1024
        # P5 = torch.cat([P4_downsample, P5], 1)
        # # 20, 20, 1024 => 20, 20, 512
        # P5 = self.conv3_for_downsample2(P5)
        # #------------------------加强特征提取网络------------------------# 
        # # P3 80, 80, 128 
        # # P4 40, 40, 256
        # # P5 20, 20, 512
        
        # P3 = self.rep_conv_1(P3)
        # P4 = self.rep_conv_2(P4)
        # P5 = self.rep_conv_3(P5)
        # #---------------------------------------------------#
        # #   第三个特征层
        # #   y3=(batch_size, 75, 80, 80)
        # #---------------------------------------------------#
        # out2 = self.yolo_head_P3(P3)
        # #---------------------------------------------------#
        # #   第二个特征层
        # #   y2=(batch_size, 75, 40, 40)
        # #---------------------------------------------------#
        # out1 = self.yolo_head_P4(P4)
        # #---------------------------------------------------#
        # #   第一个特征层
        # #   y1=(batch_size, 75, 20, 20)
        # #---------------------------------------------------#
        # out0 = self.yolo_head_P5(P5)

        return [out0, out1, out2]
