
from network.backbone.swinunet import *

import itertools
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
import network as nt
import os

from network.loss.transwarp_contrastive_loss import boundary_aware_contrast
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from network.module.imagefilter3d import GINGroupConv3D
from data.mytransforms import MyHistogramEqualizationTransform


device = torch.device("cuda:0")

class TSNetTrainer(nn.Module):
    def __init__(self):
        super(TSNetTrainer, self).__init__()

        # General settings
        self.device = torch.device('cuda:0')
        self.iters = 1
        self.ema_decay = 0.999
        self.ema_decay_conflict = 0.999
        self.ema_decay_invariant = 0.9
        self.epoch = 1

        self.data_dir = 'C:/lfm/code/monai/DomainGeneralization/'

        # Networks
        self.net_student = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=3, feature_size=24,
                                     use_checkpoint=True).to(device=self.device)

        self.net_teacher = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=3, feature_size=24,
                                     use_checkpoint=True).to(device=self.device)
        self.detach_model(self.net_teacher)
        self.nets = ['net_student', 'net_teacher']

        # Loss functions
        self.criterionDice = DiceCELoss(to_onehot_y=True, softmax=True)
        self.criterionMSE = nn.MSELoss()

        # Optimizer and Schedulers
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.net_student.parameters(), self.net_teacher.parameters()),
            lr=0.0001, betas=(0.9, 0.999))
        self.optimizers = [self.optimizer]
        self.schedulers = [nt.get_scheduler(optimizer, lr_policy='lambda') for optimizer in self.optimizers]

        # Data variables
        self.img_stu_S = None
        self.label_S = None
        self.img_tea_T = None
        self.label_T = None

        self.img_tea_S = None
        self.img_stu_T = None

        self.Histogram = MyHistogramEqualizationTransform()

        self.gingroupconv3d = GINGroupConv3D()

    def set_input(self, inputs):
        """ This function is for one-stream setup. """

        self.img_stu_S = inputs[0].to(self.device)
        self.img_tea_S = inputs[0].to(self.device)
        self.label_S = inputs[1].to(self.device)

        self.img_tea_T = self.gingroupconv3d(inputs[2].to(self.device))
        self.img_stu_T = self.gingroupconv3d(inputs[2].to(self.device))
        self.label_T = inputs[3].to(self.device)


    def forward(self):
        self.seg_stu_S, self.latent_stu_S = self.net_student(self.img_stu_S)
        self.seg_stu_T, self.latent_stu_T = self.net_student(self.img_stu_T)
        self.seg_tea_T, self.latent_tea_T = self.net_teacher(self.img_tea_T)
        self.seg_tea_S, self.latent_tea_S = self.net_teacher(self.img_tea_S)

        return self.seg_stu_S, self.seg_tea_T, self.seg_stu_T, self.seg_tea_S, \
            self.latent_stu_S, self.latent_tea_T, self.latent_stu_T, self.latent_tea_S

    def backward(self):

        # Supervised loss
        self.loss_supervised_stu_S = self.criterionDice(self.seg_stu_S, self.label_S)
        self.loss_supervised_stu_T = self.criterionDice(self.seg_stu_T, self.label_T)
        self.loss_supervised_tea_T = self.criterionDice(self.seg_tea_T, self.label_T)
        self.loss_supervised_tea_S = self.criterionDice(self.seg_tea_S, self.label_S)
        self.loss_supervised = 0.25 * self.loss_supervised_stu_S + \
                               0.25 * self.loss_supervised_stu_T + \
                               0.25 * self.loss_supervised_tea_T + \
                               0.25 * self.loss_supervised_tea_S

        self.loss_contrastive, \
            self.loss_positive1, \
            self.loss_positive2, \
            self.loss_negative1, \
            self.loss_negative2, \
            self.loss_negative3, \
            self.loss_negative4 = boundary_aware_contrast(self.latent_stu_S, self.latent_stu_T,
                                                     self.latent_tea_S, self.latent_tea_T, tao=1)

        loss = 0.8 * self.loss_supervised + \
               0.2 * self.loss_contrastive

        # Supervised loss items
        self.v_loss_supervised_stu_S = self.loss_supervised_stu_S.item()
        self.v_loss_supervised_stu_T = self.loss_supervised_stu_T.item()
        self.v_loss_supervised_tea_T = self.loss_supervised_tea_T.item()
        self.v_loss_supervised_tea_S = self.loss_supervised_tea_S.item()
        self.v_loss_supervised = self.loss_supervised.item()

        # contrastive loss
        self.v_loss_contrastive = self.loss_contrastive.item()
        self.v_loss_positive1 = self.loss_positive1.item()
        self.v_loss_positive2 = self.loss_positive2.item()
        self.v_loss_negative1 = self.loss_negative1.item()
        self.v_loss_negative2 = self.loss_negative2.item()
        self.v_loss_negative3 = self.loss_negative3.item()
        self.v_loss_negative4 = self.loss_negative4.item()

        self.v_loss = loss.item()

        loss.backward(retain_graph=True)

    @torch.no_grad()
    def ema(self):
        alpha = min(1 - 1 / (self.iters + 1), self.ema_decay)
        for ema_param, param in zip(self.net_teacher.parameters(), self.net_student.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    @torch.no_grad()
    def ema_invariant(self):
        alpha = min(1 - 1 / (self.iters + 1), self.ema_decay_invariant)
        for ema_param, param in zip(self.net_teacher.parameters(), self.net_student.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    @torch.no_grad()
    def ema_conflict(self):
        alpha = min(1 - 1 / (self.iters + 1), self.ema_decay_conflict)
        for ema_param, param in zip(self.net_teacher.parameters(), self.net_student.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


    def optimize_parameters_benchmark(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        self.ema()
        self.iters += 1

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

        self.optimizer.zero_grad()
        self.loss_supervised_stu_S.backward(retain_graph=True)
        supervised_stu_S_gradient = self.net_student.get_gradient()

        self.optimizer.zero_grad()
        self.loss_supervised_stu_T.backward(retain_graph=True)
        supervised_stu_T_gradient = self.net_student.get_gradient()

        aggregated_supervised_stu_S_gradient = torch.cat(supervised_stu_S_gradient)
        aggregated_supervised_stu_T_gradient = torch.cat(supervised_stu_T_gradient)

        signed_supervised_stu_S_gradient = torch.sign(aggregated_supervised_stu_S_gradient)
        signed_supervised_stu_T_gradient = torch.sign(aggregated_supervised_stu_T_gradient)

        cosine_similarity = F.cosine_similarity(signed_supervised_stu_S_gradient, signed_supervised_stu_T_gradient, dim=0)

        # GS-EMA, GSEMA, Gradient surgery exponential moving average
        if cosine_similarity > 0:
            self.ema_invariant()
        else:
            self.ema_conflict()

        self.iters += 1

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_loss(self):
        loss = [
            self.v_loss_supervised_stu_S,
            self.v_loss_supervised_stu_T,
            self.v_loss_supervised_tea_T,
            self.v_loss_supervised_tea_S,
            self.v_loss_supervised,
            self.v_loss_contrastive,
            self.v_loss_positive1,
            self.v_loss_positive2,
            self.v_loss_negative1,
            self.v_loss_negative2,
            self.v_loss_negative3,
            self.v_loss_negative4,
            self.v_loss
        ]
        return np.array([v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in loss])

    @staticmethod
    def detach_model(net):
        for param in net.parameters():
            param.detach_()

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

    def _eval(self):
        self.net_student.eval()
        self.eval()

    def _train(self):
        self.net_student.train()
        self.train()
