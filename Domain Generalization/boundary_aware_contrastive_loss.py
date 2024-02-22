import torch
import torch.nn as nn

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

Similarity = nn.CosineSimilarity(dim=-1)

cosine_loss = nn.CosineEmbeddingLoss(margin=0.5)

def boundary_aware_contrast(latent_stu_S, latent_stu_T, latent_tea_S, latent_tea_T, tao=1):
    bound_stu_S = get_high_freq(latent_stu_S, device)
    bound_stu_T = get_high_freq(latent_stu_T, device)
    bound_tea_S = get_high_freq(latent_tea_S, device)
    bound_tea_T = get_high_freq(latent_tea_T, device)

    positive1 = Similarity(latent_stu_S.reshape(latent_stu_S.size(0), -1), latent_tea_S.reshape(latent_tea_S.size(0), -1)).mean()
    positive2 = Similarity(latent_stu_T.reshape(latent_stu_T.size(0), -1), latent_tea_T.reshape(latent_tea_T.size(0), -1)).mean()
    negative1 = Similarity(latent_stu_S.reshape(latent_stu_S.size(0), -1), latent_stu_T.reshape(latent_stu_T.size(0), -1)).mean()
    negative2 = Similarity(latent_tea_S.reshape(latent_tea_S.size(0), -1), latent_tea_T.reshape(latent_tea_T.size(0), -1)).mean()
    negative3 = Similarity(latent_stu_S.reshape(latent_stu_S.size(0), -1), latent_tea_T.reshape(latent_tea_T.size(0), -1)).mean()
    negative4 = Similarity(latent_stu_T.reshape(latent_stu_T.size(0), -1), latent_tea_S.reshape(latent_tea_S.size(0), -1)).mean()

    pos1 = Similarity(bound_stu_S.reshape(bound_stu_S.size(0), -1), bound_tea_S.reshape(bound_tea_S.size(0), -1)).mean()
    pos2 = Similarity(bound_stu_T.reshape(bound_stu_T.size(0), -1), bound_tea_T.reshape(bound_tea_T.size(0), -1)).mean()
    neg1 = Similarity(bound_stu_S.reshape(bound_stu_S.size(0), -1), bound_stu_T.reshape(bound_stu_T.size(0), -1)).mean()
    neg2 = Similarity(bound_tea_S.reshape(bound_tea_S.size(0), -1), bound_tea_T.reshape(bound_tea_T.size(0), -1)).mean()
    neg3 = Similarity(bound_stu_S.reshape(bound_stu_S.size(0), -1), bound_tea_T.reshape(bound_tea_T.size(0), -1)).mean()
    neg4 = Similarity(bound_stu_T.reshape(bound_stu_T.size(0), -1), bound_tea_S.reshape(bound_tea_S.size(0), -1)).mean()

    loss1 = -torch.log((torch.exp(positive1) + torch.exp(positive2)) /
                       (torch.exp(positive1) + torch.exp(positive2) +
                        torch.exp(negative1) + torch.exp(negative2) +
                        torch.exp(negative3) + torch.exp(negative4)) * tao).mean()

    loss2 = -torch.log((torch.exp(pos1) + torch.exp(pos2)) /
                      (torch.exp(pos1) + torch.exp(pos2) +
                       torch.exp(neg1) + torch.exp(neg2) +
                       torch.exp(neg3) + torch.exp(neg4)) * tao).mean()

    contrast = loss2

    return contrast, pos1, pos2, neg1, neg2, neg3, neg4





def get_high_freq(y_true, device):
    y_true = y_true.type(torch.complex64)
    f = torch.fft.fftn(y_true)
    fshift = torch.fft.fftshift(f)
    x1 = torch.ones(size=y_true.shape).to(device)
    # here, the patch size is [B,N,64,64,64] so the window size is 64//4=16, from 8 to 24
    x1[:, :, 8: 24, 8: 24, 8: 24] = 0
    x1 = x1.to(torch.complex64)

    fshift = torch.mul(fshift, x1)
    ishift = torch.fft.ifftshift(fshift)
    himg = torch.fft.ifftn(ishift)
    y_true2 = torch.abs(himg)

    return y_true2


def loss(self, y_true, y_pred):
    loss = self.dice_loss(y_true, y_pred) + self.MSE_loss(y_true, y_pred)
    return loss