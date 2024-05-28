import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from network.loss.loss import DiceLoss, WeightedDiceCELoss
import network as nt
import numpy as np
import os
from data.augmentation import augmentation_fda, augmentation_fast
import skimage.filters as filters
from network.loss.loss_contrastive_loss_3D import contrastive_loss
from network.loss.transwarp_contrastive_loss import instance_contrast, style_contrast, transwarp_contrast

from torch.nn import TransformerEncoder, TransformerEncoderLayer


# from monai.networks.nets import SwinUNETR
from network.loss.loss import dice_coefficient_3d_mra_aneurysm, dice_coefficient_3d_ra_aneurysm, \
    dice_coefficient_3d_ra_vessel
import monai.transforms as transforms

from data.mytransforms import MyHistogramEqualizationTransform

device = torch.device("cuda:0")




class MultiHeadAttentionCombine_V1(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(d_model, num_heads)

    def forward(self, latent1, latent2):
        # Assuming latent1 and latent2 have the same dimensions for simplicity
        # [batch_size, sequence_length, features] is the typical input shape for nn.MultiheadAttention

        # Transpose latent1 and latent2 for multihead attention (needs seq_len, batch, features)
        latent1 = latent1.transpose(0, 1)
        latent2 = latent2.transpose(0, 1)

        # Use latent2 as query, latent1 as key and value
        attn_output, _ = self.multihead_attention(latent2, latent1, latent1)

        # Transpose back to original shape (batch, seq_len, features)
        combined_latent = attn_output.transpose(0, 1)

        # You can also consider adding a feed-forward network here
        # combined_latent = self.feed_forward(combined_latent)

        return combined_latent

import torch.nn as nn

class MultiHeadAttentionCombine_v2(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttentionCombine_v2, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, key_value, query):
        # Assuming key_value and query have shape [batch, channels, depth, height, width]
        # Flatten the spatial dimensions for multihead attention
        batch_size, channels, depth, height, width = key_value.size()
        key_value_flat = key_value.flatten(2).transpose(0, 1)  # [depth*height*width, batch, channels]
        query_flat = query.flatten(2).transpose(0, 1)          # [depth*height*width, batch, channels]

        # Apply multihead attention with query as latent2 and key/value as latent1
        attn_output, attn_output_weights = self.multihead_attn(query=query_flat, key=key_value_flat, value=key_value_flat)

        # Reshape attn_output back to 5D shape
        attn_output_5d = attn_output.transpose(0, 1).view(batch_size, channels, depth, height, width)

        return attn_output_5d


class MultiHeadAttentionCombine(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, key_value, query):
        # Flatten the spatial dimensions for multihead attention
        batch_size, channels, depth, height, width = key_value.size()
        # The expected shape for multihead attention is (sequence_length, batch_size, embed_dim)
        # where sequence_length is the product of the spatial dimensions.
        sequence_length = depth * height * width
        key_value_flat = key_value.view(batch_size, channels, sequence_length).transpose(1, 2)  # [sequence_length, batch, channels]
        query_flat = query.view(batch_size, channels, sequence_length).transpose(1, 2)  # [sequence_length, batch, channels]

        # Apply multihead attention with query as latent2 and key/value as latent1
        attn_output, attn_output_weights = self.multihead_attn(query=query_flat, key=key_value_flat, value=key_value_flat)

        # Reshape attn_output back to the original shape
        attn_output_reshaped = attn_output.transpose(1, 2).view(batch_size, channels, depth, height, width)

        return attn_output_reshaped






class TransformerLayer(nn.Module):
    def __init__(self, features, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(features, num_heads, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(features, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, features)

        self.norm1 = nn.LayerNorm(features)
        self.norm2 = nn.LayerNorm(features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.norm1(src)
        q = k = v = src2
        src2 = self.self_attn(q, k, v, attn_mask=None, key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src



class TSNetTrainer(nn.Module):
    def __init__(self):
        super(TSNetTrainer, self).__init__()

        # general settings
        self.device = torch.device('cuda:0')
        self.iters = 1
        self.ema_decay = 0.999
        self.epoch = 1

        self.ckpt_dir = '/weight/'
        self.data_dir = 'C:/lfm/code/monai/domain/'

        self.net_student = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=2, feature_size=24,
                                     use_checkpoint=True).to(device=torch.device("cuda"))
        self.net_teacher = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=2, feature_size=24,
                                     use_checkpoint=True).to(device=torch.device("cuda"))
        self.detach_model(self.net_teacher)
        self.nets = ['net_student', 'net_teacher']
        self.criterionDice = DiceCELoss(to_onehot_y=True, softmax=True)

        # weight1 = torch.from_numpy(np.array([0.01, 0.19, 0.8])).to(self.device)
        # weight1 = weight1.type(torch.FloatTensor)
        # weight2 = torch.from_numpy(np.array([0.01, 0.19, 0.8])).to(self.device)
        # weight2 = weight2.type(torch.FloatTensor)
        # self.criterionDice = WeightedDiceCELoss(to_onehot_y=True, softmax=True, dice_weight=weight1, ce_weight=weight2)

        self.criterionDice2 = DiceLoss(to_onehot_y=True, softmax=True)

        self.criterionMSE = nn.MSELoss()
        self.criterionSimilarity = nn.CosineSimilarity(dim=-1)
        # self.criterionCons = nn.KLDivLoss(reduction='batchmean')
        self.criterionCons = contrastive_loss



        # initialize optimizer
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.net_student.parameters(), self.net_teacher.parameters()),
            lr=0.0003, betas=(0.9, 0.999))
        self.optimizers = [self.optimizer]
        self.schedulers = [nt.get_scheduler(optimizer, lr_policy='lambda') for optimizer in self.optimizers]

        # data variables
        # source image (3DRA)
        self.img_stu_S = None
        self.label_stu_S = None
        self.seg_stu_S = None
        self.label_tea_T = None
        self.proj_stu_S = None
        self.pred_stu_S = None

        # source-like target image (3DRA' from mra)
        self.img_stu_T = None
        # self.label_stu_T = None # No Label
        self.seg_stu_T = None
        self.proj_stu_T = None
        self.pred_stu_T = None

        # target image (MRA)
        self.img_tea_T = None
        # self.label_tea_T = None  # No Label
        self.seg_tea_T = None
        self.proj_tea_T = None
        self.pred_tea_T = None

        # target-like source image (MRA' from 3dra)
        self.img_tea_S = None
        self.label_tea_S = None
        self.seg_tea_S = None
        self.proj_tea_S = None
        self.pred_tea_S = None

        # pseudo label
        self.pseudo_stu_S = None
        self.pseudo_stu_T = None
        self.pseudo_tea_S = None
        self.pseudo_tea_T = None

        self.latent_stu_S = None
        self.latent_stu_T = None
        self.latent_tea_T = None
        self.latent_tea_S = None

        self.gram_stu_S = None
        self.gram_stu_T = None
        self.gram_tea_T = None
        self.gram_tea_S = None

        self.enc0_stu_S = None
        self.enc0_stu_T = None
        self.enc0_tea_T = None
        self.enc0_tea_S = None

        self.enc1_stu_S = None
        self.enc1_stu_T = None
        self.enc1_tea_T = None
        self.enc1_tea_S = None

        self.enc2_stu_S = None
        self.enc2_stu_T = None
        self.enc2_tea_T = None
        self.enc2_tea_S = None

        self.enc3_stu_S = None
        self.enc3_stu_T = None
        self.enc3_tea_T = None
        self.enc3_tea_S = None

        # loss tensors
        self.loss_supervised_loss_student = None
        self.loss_supervised_loss_teacher = None
        self.loss_supervised = None

        self.loss_semi_mse_loss_source = None
        self.loss_semi_mse_loss_target = None
        self.loss_semi_sim_loss_source = None
        self.loss_semi_sim_loss_target = None
        self.loss_semi = None

        self.loss_contrastive_instance = None
        self.loss_contrastive_style = None
        self.loss_contrastive_transwarp = None

        self.loss_positive1 = None
        self.loss_positive2 = None
        self.loss_positive3 = None
        self.loss_negative1 = None
        self.loss_negative2 = None
        self.loss_negative3 = None

        self.self_supervised_loss_student_MRA = None

        # loss variables for monitor
        self.v_loss_supervised_loss_student = None
        self.v_loss_supervised_loss_teacher = None
        self.v_loss_supervised_loss_student_MRA = None
        self.v_loss_supervised = None
        self.v_loss_semi_mse_loss_source = None
        self.v_loss_semi_mse_loss_target = None
        self.v_loss_semi_sim_loss_source = None
        self.v_loss_semi_sim_loss_target = None
        self.v_loss_semi = None
        self.v_loss_contrastive_instance = None
        self.v_loss_contrastive_style = None
        self.v_loss_contrastive_transwarp = None
        self.v_loss_positive1 = None
        self.v_loss_positive2 = None
        self.v_loss_negative1 = None
        self.v_loss_negative2 = None
        self.v_loss_positive3 = None
        self.v_loss_positive4 = None
        self.v_loss_positive5 = None
        self.v_loss_positive6 = None
        self.v_loss = None

    def set_input(self, inputs):
        """ This function is for one-stream setup. """

        self.img_stu_S = inputs[0].to(self.device)
        self.label_stu_S = inputs[1].to(self.device)

        img_stu_T_numpy = torch.from_numpy(augmentation_fda(im_src=inputs[2], im_trg=inputs[0]))
        img_tea_S_numpy = torch.from_numpy(augmentation_fda(im_src=inputs[0], im_trg=inputs[2]))
        self.img_stu_T = img_stu_T_numpy.to(self.device, dtype=torch.float)
        self.img_tea_T = inputs[2].to(self.device)

        self.img_tea_S = img_tea_S_numpy.to(self.device, dtype=torch.float)
        self.label_tea_S = inputs[1].to(self.device)

        self.pseudo_stu_T = inputs[3].to(self.device)
        self.pseudo_tea_T = inputs[3].to(self.device)

    def forward(self):

        self.seg_stu_S, self.latent_stu_S, self.gram_stu_S, \
            self.enc0_stu_S, self.enc1_stu_S = self.net_student(self.img_stu_S)

        self.seg_stu_T, self.latent_stu_T, self.gram_stu_T, \
            self.enc0_stu_T, self.enc1_stu_T = self.net_student(self.img_stu_T)

        self.seg_tea_T, self.latent_tea_T, self.gram_tea_T, \
            self.enc0_tea_T, self.enc1_tea_T = self.net_teacher(self.img_tea_T)

        self.seg_tea_S, self.latent_tea_S, self.gram_tea_S, \
            self.enc0_tea_S, self.enc1_tea_S = self.net_teacher(self.img_tea_S)

        return self.seg_stu_S, self.seg_tea_T, self.seg_stu_T, self.seg_tea_S, \
            self.latent_stu_S, self.latent_tea_T, self.latent_stu_T, self.latent_tea_S, \
            self.gram_stu_S, self.gram_tea_T, self.gram_stu_T, self.gram_tea_S

    def backward(self):

        # Supervised loss
        self.loss_supervised_loss_student = self.criterionDice(self.seg_stu_S, self.label_stu_S)
        self.self_supervised_loss_student_MRA = self.criterionDice2(self.seg_stu_T, self.pseudo_stu_T)
        self.loss_supervised = self.loss_supervised_loss_student
        print('\n', 'self.loss_supervised_loss_student:', self.loss_supervised_loss_student)
        print('self.self_supervised_loss_student_MRA:', self.self_supervised_loss_student_MRA)
        print('self.loss_supervised:', self.loss_supervised)

        self.loss_semi_mse_loss_source = self.criterionMSE(F.softmax(self.seg_tea_S, dim=1).detach(),
                                                           F.softmax(self.seg_stu_S, dim=1))
        self.loss_semi_mse_loss_target = self.criterionMSE(F.softmax(self.seg_tea_T, dim=1).detach(),
                                                           F.softmax(self.seg_stu_T, dim=1))
        self.loss_semi_mse_loss_target = self.criterionMSE(self.seg_tea_T, self.seg_stu_T)
        self.loss_semi_sim_loss_source = \
            1 - self.criterionSimilarity(self.seg_tea_S.view(self.seg_tea_S.size(0), -1),
                                         self.seg_stu_S.view(self.seg_stu_S.size(0), -1)).mean()
        self.loss_semi_sim_loss_target = \
            1 - self.criterionSimilarity(self.seg_tea_T.view(self.seg_tea_T.size(0), -1),
                                         self.seg_stu_T.view(self.seg_stu_T.size(0), -1)).mean()
        self.loss_semi = 0.25 * self.loss_semi_mse_loss_source + \
                         0.25 * self.loss_semi_mse_loss_target + \
                         0.25 * self.loss_semi_sim_loss_source + \
                         0.25 * self.loss_semi_sim_loss_target

        print('self.loss_semi_mse_loss_source:', self.loss_semi_mse_loss_source)
        print('self.loss_semi_mse_loss_target:', self.loss_semi_mse_loss_target)
        print('self.loss_semi_sim_loss_source:', self.loss_semi_sim_loss_source)
        print('self.loss_semi_sim_loss_target:', self.loss_semi_sim_loss_target)
        print('self.loss_semi:', self.loss_semi)

        self.loss_contrastive_transwarp, \
            self.loss_positive1, \
            self.loss_positive2, \
            self.loss_negative1, \
            self.loss_negative2, \
            self.loss_positive3, \
            self.loss_positive4, \
            self.loss_positive5, \
            self.loss_positive6 = transwarp_contrast(self.latent_stu_S, self.latent_stu_T,
                                                     self.latent_tea_S, self.latent_tea_T,
                                                     self.gram_stu_S, self.gram_stu_T,
                                                     self.gram_tea_S, self.gram_tea_T, tao=1)
        print('self.loss_contrastive_transwarp:', self.loss_contrastive_transwarp)

        loss = 0.8 * self.loss_supervised + \
               0.1 * self.loss_semi + \
               0.1 * self.loss_contrastive_transwarp

        # Supervised loss
        self.v_loss_supervised_loss_student = self.loss_supervised_loss_student.item()
        self.v_loss_supervised_loss_student_MRA = self.self_supervised_loss_student_MRA.item()
        self.v_loss_supervised = self.loss_supervised.item()

        # semi supervised loss
        self.v_loss_semi_mse_loss_source = self.loss_semi_mse_loss_source.item()
        self.v_loss_semi_mse_loss_target = self.loss_semi_mse_loss_target.item()
        self.v_loss_semi_sim_loss_source = self.loss_semi_sim_loss_source.item()
        self.v_loss_semi_sim_loss_target = self.loss_semi_sim_loss_target.item()
        self.v_loss_semi = self.loss_semi

        # Transwrap contrastive loss
        self.v_loss_contrastive_instance = None
        self.v_loss_contrastive_style = None
        self.v_loss_contrastive_transwarp = self.loss_contrastive_transwarp.item()

        self.v_loss_positive1 = self.loss_positive1.item()
        self.v_loss_positive2 = self.loss_positive2.item()
        self.v_loss_negative1 = self.loss_negative1.item()
        self.v_loss_negative2 = self.loss_negative2.item()
        self.v_loss_positive3 = self.loss_positive3.item()
        self.v_loss_positive4 = self.loss_positive4.item()
        self.v_loss_positive5 = self.loss_positive5.item()
        self.v_loss_positive6 = self.loss_positive6.item()

        self.v_loss = loss.item()

        loss.backward()

    @torch.no_grad()
    def ema(self):
        alpha = min(1 - 1 / (self.iters + 1), self.ema_decay)
        for ema_param, param in zip(self.net_teacher.parameters(), self.net_student.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        self.ema()
        # self.update_centroid()
        # self.update_pixel()
        self.iters += 1

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_loss(self):
        loss = [
            self.v_loss_supervised_loss_student,
            self.v_loss_supervised_loss_student_MRA,
            self.v_loss_supervised,
            self.v_loss_semi_mse_loss_source,
            self.v_loss_semi_mse_loss_target,
            self.v_loss_semi_sim_loss_source,
            self.v_loss_semi_sim_loss_target,
            self.v_loss_semi,
            self.v_loss_contrastive_instance,
            self.v_loss_contrastive_style,
            self.v_loss_contrastive_transwarp,
            self.v_loss_positive1,
            self.v_loss_positive2,
            self.v_loss_negative1,
            self.v_loss_negative2,
            self.v_loss_positive3,
            self.v_loss_positive4,
            self.v_loss_positive5,
            self.v_loss_positive6,
            self.v_loss
        ]
        # return np.array(loss).cpu()
        # return np.array([v.cpu() for v in loss])
        # return np.array([v.cpu() if isinstance(v, torch.Tensor) else v for v in loss])
        return np.array([v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in loss])

    @staticmethod
    def detach_model(net):
        for param in net.parameters():
            param.detach_()

    @staticmethod
    def get_pseudo_anchor_label(image):
        # Convert the tensor to a NumPy array
        image_np = image.cpu().numpy()

        # AHA
        # 创建直方图归一化的转换函数
        Normalize = transforms.NormalizeIntensity()
        scaler = transforms.ScaleIntensity(minv=0, maxv=1)
        Histogram = MyHistogramEqualizationTransform()
        # 对图像进行直方图归一化处理
        image_np = Normalize(image_np)
        image_np = Histogram(image_np)
        image_np = scaler(image_np)
        # 将处理后的图像保存为.nii.gz文件
        image_np = image_np.cpu().numpy()

        # Apply thresholding using scikit-image
        threshold = filters.threshold_otsu(image_np)
        binary_image_np = image_np > threshold
        binary_image_np = binary_image_np.astype(np.uint8)

        # Convert the NumPy array back to a PyTorch tensor
        pseudo_label = torch.from_numpy(binary_image_np)
        # print(np.sum(binary_image_np), 'np.sum(binary_image_np)'')

        # Return the binary image as a PyTorch tensor
        return pseudo_label

    def save_networks(self, save_name=None):
        for net_name in self.nets:
            net = self.__getattr__(net_name)
            if save_name:
                save_filename = '{}_{}_iter{}.pth'.format(net_name, net.module.name, save_name)
            else:
                save_filename = '{}_{}_iter{}.pth'.format(net_name, net.module.name, self.epoch)
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)
            save_path = os.path.join(self.data_dir, save_filename)
            print('save path:', save_path)

            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()
            torch.save({'state_dict': state_dict, 'epoch': self.epoch}, save_path)

    # def load_networks(self):

    def _eval(self):
        self.net_student.eval()
        self.eval()

    def _train(self):
        self.net_student.train()
        self.train()


class Transformer(nn.Module):  # Dual Attention + 3-Transformer

    def __init__(self):
        super(Transformer, self).__init__()
        self.attention = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                                        dim_feedforward=2048, dropout=0.1, activation=F.relu, )

    def forward(self, x):
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.transformer = TransformerEncoder(TransformerEncoderLayer(d_model, nhead), num_layers=num_layers)
        self.transformer = self.transformer.to(device)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.transformer(x).to(device)

from typing import Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, optional_import

rearrange, _ = optional_import("einops", name="rearrange")


# Define the diffusion process
class Diffusion(nn.Module):
    def __init__(self, num_iterations, dt, alpha):
        super(Diffusion, self).__init__()
        self.num_iterations = num_iterations
        self.dt = dt
        self.alpha = alpha

    def forward(self, x):
        for i in range(self.num_iterations):
            # print(x.shape, 'ccccccccccc x') # ([4, 3, 128, 128, 128])
            x_grad = torch.gradient(x)
            # print(x.shape, 'ccccccccccc x') # ([4, 3, 128, 128, 128])
            x_grad = torch.stack(x_grad)
            laplacian = x_grad.sum(dim=0)
            # print(laplacian.shape, 'ccccccccccc laplacian') # ([4, 3, 128, 128, 128])
            x = x + self.alpha * self.dt * laplacian
        return x


class SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
            self,
            img_size: Union[Sequence[int], int],
            in_channels: int,
            out_channels: int,
            depths: Sequence[int] = (2, 2),
            num_heads: Sequence[int] = (3, 6),
            feature_size: int = 24,
            norm_name: Union[Tuple, str] = "instance",
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            dropout_path_rate: float = 0.0,
            normalize: bool = True,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.diffusion = Diffusion(num_iterations=10, dt=0.1, alpha=0.5)

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # self.encoder10 = UnetrBasicBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=16 * feature_size,
        #     out_channels=16 * feature_size,
        #     kernel_size=3,
        #     stride=1,
        #     norm_name=norm_name,
        #     res_block=True,
        # )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        self.proj_head = ProjectionHead(384, 128)

        self.pred_head = ProjectionHead(128, 128)

        self.attn = nn.MultiheadAttention(embed_dim=48, num_heads=8, dropout=0.1)

        self.combine_layer = MultiHeadAttentionCombine(embed_dim=48, num_heads=8, dropout=0.1)

    @staticmethod
    def function1(feature):
        # batch, channels, depth, height, width = feature.size()
        # feature = feature.view(batch, channels, -1)
        feature = feature.to(device)
        feature = feature.permute(0, 2, 1)
        gram = torch.bmm(feature, feature.transpose(1, 2)).to(device)  # Compute the Gram matrix
        # print(gram.shape, 'aaaaaaa gram.shape')
        return gram

    @staticmethod
    def function2(gram, original_feature):
        batch, channels, depth, height, width = original_feature.size()

        transformer_block = TransformerBlock(channels, 8, 1)

        original_feature = original_feature.view(batch, channels, -1)

        value = transformer_block(original_feature).to(device)
        # value = value.view(batch, channels, depth*height*width)
        # print(value.shape, 'dddddddd after attention value.shape')
        # print(gram.shape, 'dddddddd after attention gram.shape')
        out = torch.bmm(value, gram.permute(0, 2, 1))
        out = out.view(batch, channels, depth, height, width)

        # print(out.shape, 'aaaaaaa after attention out.shape')

        return out

    def get_low_freq_amp_np(self, src_img, L=0.5):
        # get fft of source
        fft_src_np = np.fft.fftn(src_img, axes=(-3, -2, -1))

        # extract amplitude of fft
        amp_src = np.abs(fft_src_np)

        # extract low frequency amplitude
        amp_src_ = self.low_freq_mutate_np(amp_src, L=L)

        return amp_src_

    def low_freq_mutate_np(self, amp_src, L=0.5):
        a_src = np.fft.fftshift(amp_src, axes=(-3, -2, -1))

        _, _, h, w, d = a_src.shape
        # b = (np.floor(np.amin((h, w)) * L)).astype(int)
        b = 2
        c_h = np.floor(h / 2.0).astype(int)
        c_w = np.floor(w / 2.0).astype(int)
        c_d = np.floor(d / 2.0).astype(int)

        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1
        d1 = c_d - b
        d2 = c_d + b + 1

        a_src = a_src[:, :, h1:h2, w1:w2, d1:d2]

        a_src = np.fft.ifftshift(a_src, axes=(-3, -2, -1))
        return a_src

    def get_low_freq_amp_5D(self, src_img, L=0.5):
        N, C, H, W, D = src_img.shape
        # low_freq_amp_src = torch.zeros_like(src_img)
        # low_freq_amp_src = torch.zeros((N, C, 17, 17, 17), dtype=torch.float32, device=src_img.device)
        low_freq_amp_src = torch.zeros((N, C, 5, 5, 5), dtype=torch.float32, device=src_img.device)
        src_img_np = src_img.detach().cpu().numpy()
        for i in range(C):
            channel_img = src_img_np[:, i]
            channel_img = np.expand_dims(channel_img, axis=1)
            low_freq_amp = self.get_low_freq_amp_np(channel_img, L)
            low_freq_amp_src[:, i] = torch.from_numpy(low_freq_amp).to(src_img.device)

        return low_freq_amp_src

    def load_from(self, weights):

        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    def forward_v1(self, x_in):
        x_in = x_in.to(device)
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)                  # [1, 12, 128, 128, 128]
        enc1 = self.encoder2(hidden_states_out[0])  # [1, 12, 64, 64, 64]

        latent1 = self.encoder3(hidden_states_out[1])   # [1, 24, 32, 32, 32]

        latent2 = nn.MultiheadAttention(embed_dim=24, num_heads=8, dropout=0.1)(latent1, latent1, latent1)

        low_freq_amp = self.get_low_freq_amp_5D(latent2, L=0.25)            # [1, 48, 32, 32, 32]

        combine_layer = MultiHeadAttentionCombine(d_model=latent1.size(-1), num_heads=8)
        combined_latent = combine_layer(latent1, latent2)

        dec1 = self.decoder2(combined_latent, enc1)     # latent4 [1, 24, 32, 32, 32], enc1 [1, 12, 64, 64, 64]
        dec0 = self.decoder1(dec1, enc0)
        logits = self.out(dec0)

        return logits, latent1, low_freq_amp, enc0, enc1

    def forward(self, x_in):
        x_in = x_in.to(device)
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)  # [batch_size, 12, D, H, W]
        enc1 = self.encoder2(hidden_states_out[0])  # [batch_size, 12, D/2, H/2, W/2]

        latent1 = self.encoder3(hidden_states_out[1])  # [batch_size, 24, D/4, H/4, W/4]

        # Reshape latent1 from 5D to 3D by combining all spatial dimensions into one
        # The new shape will have the form [batch_size, channels, D*H*W/16]
        latent1_flat = latent1.view(latent1.size(0), latent1.size(1), -1).transpose(1, 2)  # [batch_size, D*H*W/16, channels]

        # Apply multihead attention
        attn_output, _ = self.attn(latent1_flat, latent1_flat,
                                   latent1_flat)  # Output shape: [batch_size, D*H*W/16, channels]

        # Reshape attn_output back to the original 5D shape of latent1
        latent2 = attn_output.transpose(1, 2).view(latent1.size())  # [batch_size, channels, D/4, H/4, W/4]

        # Apply get_low_freq_amp_5D to latent2 now
        low_freq_amp = self.get_low_freq_amp_5D(latent2, L=0.25)  # [batch_size, 48, D/4, H/4, W/4]

        # Now we need to combine latent1 and latent2 using your MultiHeadAttentionCombine
        # Assuming MultiHeadAttentionCombine can handle 5D inputs, or you might need to flatten them as done before
        combined_latent = self.combine_layer(latent1,
                                             latent2)  # Assuming it returns [batch_size, channels, D/4, H/4, W/4]

        # Continue with your decoders
        dec1 = self.decoder2(combined_latent,
                             enc1)  # [batch_size, channels, D/4, H/4, W/4] and [batch_size, 12, D/2, H/2, W/2]
        dec0 = self.decoder1(dec1, enc0)  # Same as above, adjusted for the decoder's expected input
        logits = self.out(dec0)

        return logits, latent1, low_freq_amp, enc0, enc1


def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Sequence[int],
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        # qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.qkv_q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        k = self.qkv_k(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v = self.qkv_v(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        # # (1000, 343, 24) to (1000, 3, 343, 8)
        # q = self.qkv_q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        # k = self.qkv_k(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        # v = self.qkv_v(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        # qkv = torch.concat([torch.unsqueeze(q, 0), torch.unsqueeze(k, 0), torch.unsqueeze(v, 0)], dim=0)
        # print(qkv.shape, 'aaaaaaaaaaaaaaaaaaaa')
        # q, k, v = qkv[0], qkv[1], qkv[2]
        # print(q.shape, 'qqqqqqqqq')
        # print(k.shape, 'kkkkkkkk')
        # print(v.shape, 'vvvvvvvv')

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Sequence[int],
            shift_size: Sequence[int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            act_layer: str = "GELU",
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()
        x = self.norm1(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def load_from(self, weights, n_block, layer):
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
            "attn.qkv_q.weight",
            "attn.qkv_q.bias",
            "attn.qkv_k.weight",
            "attn.qkv_k.bias",
            "attn.qkv_v.weight",
            "attn.qkv_v.bias",

        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])
            self.attn.qkv_q.weight.copy_(weights["state_dict"][root + block_names[14]])
            self.attn.qkv_q.bias.copy_(weights["state_dict"][root + block_names[15]])
            self.attn.qkv_k.weight.copy_(weights["state_dict"][root + block_names[16]])
            self.attn.qkv_k.bias.copy_(weights["state_dict"][root + block_names[17]])
            self.attn.qkv_v.weight.copy_(weights["state_dict"][root + block_names[18]])
            self.attn.qkv_v.bias.copy_(weights["state_dict"][root + block_names[19]])

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMerging(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):

        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, 0::2, :]
            x3 = x[:, 0::2, 0::2, 1::2, :]
            x4 = x[:, 1::2, 0::2, 1::2, :]
            x5 = x[:, 0::2, 1::2, 0::2, :]
            x6 = x[:, 0::2, 0::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            dim: int,
            depth: int,
            num_heads: int,
            window_size: Sequence[int],
            drop_path: list,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            downsample: isinstance = None,  # type: ignore
            use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: downsample layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c d h w -> b d h w c")
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b d h w c -> b c d h w")

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c h w -> b h w c")
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x


class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
            self,
            in_chans: int,
            embed_dim: int,
            window_size: Sequence[int],
            patch_size: Sequence[int],
            depths: Sequence[int],
            num_heads: Sequence[int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            patch_norm: bool = False,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=PatchMerging,
                use_checkpoint=use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        # x2 = self.layers2[0](x1.contiguous())
        # x2_out = self.proj_out(x2, normalize)
        # x3 = self.layers3[0](x2.contiguous())
        # x3_out = self.proj_out(x3, normalize)
        # x4 = self.layers4[0](x3.contiguous())
        # x4_out = self.proj_out(x4, normalize)
        # return [x0_out, x1_out, x2_out, x3_out, x4_out]
        return [x0_out, x1_out]


class STNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(STNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.proj_head = ProjectionHead(1024, 128)
        self.pred_head = ProjectionHead(128, 128)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        proj = self.proj_head(x5)
        pred = self.pred_head(proj)

        return logits, proj, pred


class TSNetDual(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(TSNetDual, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.proj_head = ProjectionHead(1024, 256)
        self.pred_head = ProjectionHead(256, 128)

    def forward(self, x_s, x_t):
        x1_s = self.inc(x_s)
        x2_s = self.down1(x1_s)
        x3_s = self.down2(x2_s)
        x4_s = self.down3(x3_s)
        x5_s = self.down4(x4_s)
        x_s = self.up1(x5_s, x4_s)
        x_s = self.up2(x_s, x3_s)
        x_s = self.up3(x_s, x2_s)
        x_s = self.up4(x_s, x1_s)
        logits_s = self.outc(x_s)

        proj_s = self.proj_head(x5_s)
        pred_s = self.pred_head(proj_s)

        x1_t = self.inc(x_t)
        x2_t = self.down1(x1_t)
        x3_t = self.down2(x2_t)
        x4_t = self.down3(x3_t)
        x5_t = self.down4(x4_t)
        x_t = self.up1(x5_t, x4_t)
        x_t = self.up2(x_t, x3_t)
        x_t = self.up3(x_t, x2_t)
        x_t = self.up4(x_t, x1_t)
        logits_t = self.outc(x_t)

        proj_t = self.proj_head(x5_t)
        pred_t = self.pred_head(proj_t)

        return logits_s, logits_t, proj_s, pred_s, proj_t, pred_t


class ProjectionHead(nn.Module):
    def __init__(self, input_nc, output_nc, proj='convmlp'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv3d(input_nc, output_nc, kernel_size=(1, 1, 1))
        elif proj == 'convmlp':  # there is no mlp from MSCDA code
            self.proj = nn.Sequential(
                nn.Conv3d(input_nc, input_nc, kernel_size=(1, 1, 1)),
                # nn.SyncBatchNorm(input_nc),
                nn.BatchNorm3d(input_nc),
                nn.ReLU(),
                nn.Conv3d(input_nc, output_nc, kernel_size=(1, 1, 1))
            )
        # elif proj == 'mlp':

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()  #
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(784, 512)  # 第一个隐含层
        self.fc2 = torch.nn.Linear(512, 128)  # 第二个隐含层
        self.fc3 = torch.nn.Linear(128, 10)  # 输出层

    def forward(self, x):
        # 前向传播， 输入值：x, 返回值 x
        x = x.view(-1, 28 * 28)  # 将一个多行的Tensor,拼接成一行
        x = F.relu(self.fc1(x))  # 使用 relu 激活函数
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)  # 输出层使用 softmax 激活函数
        # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """ (conv => [BN] => ReLU) * 2 """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # there might be padding issue

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        # x = self.softmax(x)
        return x
