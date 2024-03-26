import torch
import torch.nn as nn

from common import SiLU, Silence, RepNCSPELAN4, Conv

class Backbone(nn.Module):
    def __init__(self, transition_channels, block_channels, n, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   transition_channels == 32
        #-----------------------------------------------#
        self.Silence = Silence()
        self.dark1 = nn.Sequential(
            # 640, 640, 3 -> 320, 320, 64
            Conv(3, transition_channels * 2, 3, 2), 
            # 320, 320, 64 -> 160, 160, 128
            Conv(transition_channels * 2, transition_channels * 4, 3, 2),
            # 160, 160, 128 -> 160, 160, 256
            RepNCSPELAN4(transition_channels * 4, transition_channels *8, 128, 64, 1)# ch_in, ch_out, number, shortcut, groups, expansion

        )
        self.dark2 = nn.Sequential(
            # 160, 160, 256 -> 80, 80, 256
            Conv(transition_channels *8, transition_channels *8, 3, 2),
            # 80, 80, 256 -> 80, 80, 512
            RepNCSPELAN4(transition_channels *8, transition_channels *16, 256, 128, 1)
        )
        self.dark3 = nn.Sequential(
            # 80, 80, 512 -> 40, 40, 512
            Conv(transition_channels *16, transition_channels *16, 3, 2),
            # 40, 40, 512 -> 40, 40, 512
            RepNCSPELAN4(transition_channels *16, transition_channels *16, 512, 256, 1)
        )
        self.dark4 = nn.Sequential(
            # 40, 40, 512 -> 20, 20, 512
            Conv(transition_channels *16, transition_channels *16, 3, 2),
            # 20, 20, 512 -> 20, 20, 512
            RepNCSPELAN4(transition_channels *16, transition_channels *16, 512, 256, 1)
        )
        self.dark5 = nn.Sequential(
            # 640, 640, 3 -> 320, 320, 64
            Conv(3, transition_channels * 2, 3, 2),
            # 320, 320, 64 -> 160 , 160, 128
            Conv(transition_channels * 2, transition_channels * 4, 3, 2),
            # 160, 160, 128 -> 160, 160, 256
            RepNCSPELAN4(transition_channels * 4, transition_channels * 8, 128, 64, 1)

        )

        # ids = {
        #     'l' : [-1, -3, -5, -6],
        #     'x' : [-1, -3, -5, -7, -8], 
        # }[phi]

        
        # if pretrained:
        #     url = {
        #         "l" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth',
        #         "x" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_x_backbone_weights.pth',
        #     }[phi]
        #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
        #     self.load_state_dict(checkpoint, strict=False)
        #     print("Load weights from " + url.split('/')[-1])

    def forward(self, x):

        x = self.Silence(x)
        #-----------------------------------------------#
        #   dark1的输出为 160, 160, 256 是一个有效特征层
        #-----------------------------------------------#
        feat1 = self.dark1(x)
        #-----------------------------------------------#
        #   dark2的输出为 80, 80, 512 是一个有效特征层
        #-----------------------------------------------#
        feat2 = self.dark2(feat1)
        #-----------------------------------------------#
        #   dark3的输出为 40, 40, 512 是一个有效特征层
        #-----------------------------------------------#
        feat3 = self.dark3(feat2)
        #-----------------------------------------------#
        #   dark4的输出为 20, 20, 512 是一个有效特征层
        #-----------------------------------------------#
        feat4 = self.dark4(feat3)


        #-----------------------------------------------#
        #   第0层tensor输入到dark5
        #   dark5的输出为 160, 160, 256 是一个有效特征层
        #-----------------------------------------------#
        feat5 = self.dark5(x)




        

        return feat1, feat2, feat3, feat4, feat5


def main():
    # sppcspc = SPPCSPC.
    img_tensor = torch.randn(1, 3, 640, 640)
    print(f"The image2tensor shape is {img_tensor.shape}.")

    backbone = Backbone(32, 32, 3, 0)
    output = backbone(img_tensor)
    print(f"The backbone shape is {output.shape}.")

if __name__ == '__main__':
    main()