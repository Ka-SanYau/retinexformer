# from torch import nn as nn
# from torch.nn import functional as F

# from basicsr.utils.registry import ARCH_REGISTRY
# from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer


# @ARCH_REGISTRY.register()
# class MSRResNet(nn.Module):
#     """Modified SRResNet.

#     A compacted version modified from SRResNet in
#     "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
#     It uses residual blocks without BN, similar to EDSR.
#     Currently, it supports x2, x3 and x4 upsampling scale factor.

#     Args:
#         num_in_ch (int): Channel number of inputs. Default: 3.
#         num_out_ch (int): Channel number of outputs. Default: 3.
#         num_feat (int): Channel number of intermediate features. Default: 64.
#         num_block (int): Block number in the body network. Default: 16.
#         upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
#     """

#     def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
#         super(MSRResNet, self).__init__()
#         self.upscale = upscale

#         self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
#         self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

#         # upsampling
#         if self.upscale in [2, 3]:
#             self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
#             self.pixel_shuffle = nn.PixelShuffle(self.upscale)
#         elif self.upscale == 4:
#             self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
#             self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
#             self.pixel_shuffle = nn.PixelShuffle(2)

#         self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

#         # activation function
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#         # initialization
#         default_init_weights([self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
#         if self.upscale == 4:
#             default_init_weights(self.upconv2, 0.1)

#     def forward(self, x):
#         feat = self.lrelu(self.conv_first(x))
#         out = self.body(feat)

#         if self.upscale == 4:
#             out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
#             out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
#         elif self.upscale in [2, 3]:
#             out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

#         out = self.conv_last(self.lrelu(self.conv_hr(out)))
#         base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
#         out += base
#         return out



from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer

import torch

global ccq_i
ccq_i = True


def get_poe_new(batch_size, height, width, rgb, channel,d_model=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # rgb = torch.tensor(rgb * 64, dtype=torch.float32).to(device)
    rgb_255 = torch.tensor((rgb * 255).byte(), dtype=torch.int32).to(device)

    # 创建位置编码张量
    pe = torch.zeros((batch_size, d_model, height, width), device=device)
    
    # 创建位置坐标
    # x = torch.linspace(1, height, steps=height, device=device).view(-1, 1).repeat(1, width)
    # y = torch.linspace(1, width, steps=width, device=device).view(1, -1).repeat(height, 1)

    for b in range(batch_size):
        for i in range(d_model):
            div_term = 10000 ** (i / d_model)
            if i % 2 == 0:
                pe[b, i] = (
                    torch.sin(rgb_255[b, channel, :, :] / div_term)
                )
            else:
                pe[b, i] = (
                    torch.cos(rgb_255[b, channel, :, :] / div_term)
                )
    
    # p1 = pe[:, 0:16].mean(dim=1, keepdim=True)  # Mean across channels 0-15
    # p2 = pe[:, 16:32].mean(dim=1, keepdim=True)  # Mean across channels 16-31
    # p3 = pe[:, 32:48].mean(dim=1, keepdim=True)  # Mean across channels 32-47
    # p4 = pe[:, 48:64].mean(dim=1, keepdim=True)   # Mean across channels 48-63

    return pe


@ARCH_REGISTRY.register()
class MSRResNet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3,                  
                 poemap=False,
                 poe_enhance=False,
                 poe_3=False,
                 hiseq=False,
                 his_ce=False,
                 num_feat=64, num_block=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.poe = poemap
        self.poe_enhance = poe_enhance
        self.poe_3 = poe_3
        self.hiseq = hiseq
        self.his_ce = his_ce
        if self.hiseq:
            num_in_ch = num_in_ch + 3

        self.conv_ce = nn.Conv2d(48, 32, 3, 1, 1)
        self.conv_down_ce = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        # self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv_first = nn.Conv2d(num_in_ch, num_feat*3, 3, 1, 1)

        self.conv_first2 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.upfea = nn.Conv2d(num_feat, num_feat*3, 3, 1, 1)
        self.ping = nn.Conv2d(num_feat*3, num_feat*3, 1, 1, 0)
        self.downfea = nn.Conv2d(num_feat*3, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 1:
            self.upconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.downconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            # self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        x_copy = x[:,:3]
        global ccq_i
        if not(self.hiseq):
            if ccq_i:
                print("not adding hiseq.")
                print(x[:,:3].shape)
            feat = self.conv_first2(x[:,:3]) # 1, 64, h, w
        else:
            if ccq_i:
                print("add hiseq.")
                print(x.shape)
            feat = self.conv_first2(x) # 1, 64, h, w

        if self.poe:
            bsize, _, h, w = x.shape
            tmp = x[:,:3]
            # color_md_1 = get_poe_new(bsize, h, w, tmp, 0, 32)
            # color_md_2 = get_poe_new(bsize, h, w, tmp, 1, 32)
            # color_md_3 = get_poe_new(bsize, h, w, tmp, 2, 32)
            color_md_1 = get_poe_new(bsize, h, w, tmp, 0, 16)
            color_md_2 = get_poe_new(bsize, h, w, tmp, 1, 16)
            color_md_3 = get_poe_new(bsize, h, w, tmp, 2, 16)

            color_md_new = torch.cat([color_md_1, color_md_2, color_md_3], dim=1)
            # tmp = torch.cat([feat, self.conv_ce(color_md_new)], dim=1)
            feat = torch.cat([feat, self.conv_ce(color_md_new)], dim=1)
            if ccq_i:
                print("self.conv_first(x) cat color_md_new.")
            # feat = self.conv_down_ce(tmp)

        if self.his_ce:
            bsize, _, h, w = x.shape
            tmp = x[:, 3:6]

            # color_md_1 = get_poe_new(bsize, h, w, tmp, 0, 32)
            # color_md_2 = get_poe_new(bsize, h, w, tmp, 1, 32)
            # color_md_3 = get_poe_new(bsize, h, w, tmp, 2, 32)
            color_md_1 = get_poe_new(bsize, h, w, tmp, 0, 16)
            color_md_2 = get_poe_new(bsize, h, w, tmp, 1, 16)
            color_md_3 = get_poe_new(bsize, h, w, tmp, 2, 16)

            color_md_new = torch.cat([color_md_1, color_md_2, color_md_3], dim=1)

            feat = torch.cat([feat, self.conv_ce(color_md_new)], dim=1)
            if ccq_i:
                print("self.conv_first(x) cat his_ce.")
            feat = self.conv_down_ce(feat)

        feat = self.lrelu(feat)

        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        elif self.upscale == 1:
            if self.poe:
                # bsize, _, h, w = x.shape
                # ce_2 = get_poe_new(bsize, h, w, x, 0, 64) + get_poe_new(bsize, h, w, x, 1, 64) + get_poe_new(bsize, h, w, x, 2, 64)
                
                # out = self.lrelu(self.lrelu(self.downconv1(self.lrelu(self.upconv1(out) + ce_2))))
                # out = self.lrelu(self.lrelu(self.downconv1(self.lrelu(self.upconv2(out) + ce_2))))
                # out = self.lrelu(self.lrelu(self.downconv1(self.upconv1(out))))
                # out = self.lrelu(self.lrelu(self.downconv1(self.lrelu(self.upconv2(out)))))
                pass
            else:
                # out = self.lrelu(self.downconv1(self.upconv1(out)))
                pass
                # out = self.lrelu(self.downconv1(self.upconv2(out)))
                # out = self.lrelu(self.lrelu(self.downconv1(self.lrelu(self.upconv2(out)))))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))

        if self.upscale == 1:
            if ccq_i:
                print(torch.mean(x))
                print("x. shape: ", x.shape)
                print("asdfaf")
            
            base = x_copy
            ccq_i = False
        else:
            base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
            # base = x_copy
        # print(out.shape)
        # print(base.shape)
        out += base
        return out



@ARCH_REGISTRY.register()
class MSRResNet_SR(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3,                  
                 poemap=False,
                 poe_enhance=False,
                 poe_3=False,
                 num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_SR, self).__init__()
        self.upscale = upscale

        self.poe = poemap
        self.poe_enhance = poe_enhance
        self.poe_3 = poe_3

        # self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.upfea = nn.Conv2d(num_feat, num_feat*3, 3, 1, 1)
        self.ping = nn.Conv2d(48, num_feat, 1, 1, 0)
        self.ce_feat_conv = nn.Conv2d(48, num_feat, 3, 1, 1)
        self.downfea = nn.Conv2d(num_feat*3, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat*2)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 1:
            self.upconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.downconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            # self.pixel_shuffle = nn.PixelShuffle(2)

        # self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat*2, num_feat*2, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat*2, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):
        x_copy = x
        global ccq_i

        if self.poe:
            bsize, _, h, w = x.shape
            color_md_1 = get_poe_new(bsize, h, w, x, 0, 16)
            color_md_2 = get_poe_new(bsize, h, w, x, 1, 16)
            color_md_3 = get_poe_new(bsize, h, w, x, 2, 16)

            # color_md_1 = torch.mean(color_md_1, dim=1, keepdim=True)
            # color_md_2 = torch.mean(color_md_2, dim=1, keepdim=True)
            # color_md_3 = torch.mean(color_md_3, dim=1, keepdim=True)
            color_md_new = torch.cat([color_md_1, color_md_2, color_md_3], dim=1)
            # color_md_new = color_md_1 + color_md_2 + color_md_3
            # color_md_new = color_md_new * self.img_range
            # x = torch.cat([x, color_md_new], dim=1)
            # x += color_md_new
            
        if self.poe:
            if ccq_i:
                print("self.conv_first(x) + color_md_new.")
            # feat = self.lrelu(self.downfea(self.lrelu(self.conv_first(x))))
            # feat = self.conv_first(x) + color_md_new
            # feat = self.lrelu(feat)
            # feat = self.lrelu(self.downfea(self.conv_first(x) + color_md_new))
            feat = self.conv_first(x)
            ce_feat = self.ce_feat_conv(color_md_new)

            feat = torch.cat([feat, ce_feat], dim=1)

            feat = self.lrelu(feat)
            # feat = self.upfea(feat) 
            # feat = feat + self.ping(color_md_new)
            # feat = self.lrelu(self.downfea(feat))

        else:
            feat = self.lrelu(self.downfea(self.conv_first(x)))

        # feat += color_md_new

        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        elif self.upscale == 1:
            if self.poe:
                # bsize, _, h, w = x.shape
                # ce_2 = get_poe_new(bsize, h, w, x, 0, 64) + get_poe_new(bsize, h, w, x, 1, 64) + get_poe_new(bsize, h, w, x, 2, 64)
                
                # out = self.lrelu(self.lrelu(self.downconv1(self.lrelu(self.upconv1(out) + ce_2))))
                # out = self.lrelu(self.lrelu(self.downconv1(self.lrelu(self.upconv2(out) + ce_2))))
                # out = self.lrelu(self.lrelu(self.downconv1(self.upconv1(out))))
                # out = self.lrelu(self.lrelu(self.downconv1(self.lrelu(self.upconv2(out)))))
                pass
            else:
                # out = self.lrelu(self.downconv1(self.upconv1(out)))
                pass
                # out = self.lrelu(self.downconv1(self.upconv2(out)))
                # out = self.lrelu(self.lrelu(self.downconv1(self.lrelu(self.upconv2(out)))))



        out = self.conv_last(self.lrelu(self.conv_hr(out)))


        if self.upscale == 1:
            if ccq_i:
                print(torch.mean(x))
                print("x. shape: ", x.shape)
                print("asdfaf")
            
            base = x_copy
            ccq_i = False
        else:
            base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
            # base = x_copy
        # print(out.shape)
        # print(base.shape)
        out += base
        return out




@ARCH_REGISTRY.register()
class MSRResNet_Real_SR(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3,                  
                 poemap=False,
                 poe_enhance=False,
                 poe_3=False,
                 num_feat=64, num_block=16, upscale=4):
        super(MSRResNet_Real_SR, self).__init__()
        self.upscale = upscale

        self.poe = poemap
        self.poe_enhance = poe_enhance
        self.poe_3 = poe_3

        self.upscale = upscale

        self.conv_ce = nn.Conv2d(48, num_feat, 3, 1, 1)
        self.conv_down_ce = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x):

        
        feat = self.conv_first(x)
        global ccq_i


        if self.poe:
            bsize, _, h, w = x.shape
            color_md_1 = get_poe_new(bsize, h, w, x, 0, 16)
            color_md_2 = get_poe_new(bsize, h, w, x, 1, 16)
            color_md_3 = get_poe_new(bsize, h, w, x, 2, 16)
            
            if ccq_i:
                print("aasdfasdf")
                ccq_i = False
            # color_md_1 = torch.mean(color_md_1, dim=1, keepdim=True)
            # color_md_2 = torch.mean(color_md_2, dim=1, keepdim=True)
            # color_md_3 = torch.mean(color_md_3, dim=1, keepdim=True)
            color_md_new = torch.cat([color_md_1, color_md_2, color_md_3], dim=1)
            # color_md_new = color_md_1 + color_md_2 + color_md_3
            # color_md_new = color_md_new * self.img_range
            # x = torch.cat([x, color_md_new], dim=1)
            # x += color_md_new
            tmp = torch.cat([feat, self.conv_ce(color_md_new)], dim=1)
            feat = self.conv_down_ce(tmp)


        feat = self.lrelu(feat)
        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))



        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        out += base
        return out