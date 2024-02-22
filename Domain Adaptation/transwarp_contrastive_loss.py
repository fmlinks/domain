import torch
import torch.nn as nn

Similarity = nn.CosineSimilarity(dim=-1)

cosine_loss = nn.CosineEmbeddingLoss(margin=0.5)


def instance_contrast(stu_S, stu_T, tea_S, tea_T, tao=1):

    positive1 = Similarity(stu_S.view(stu_S.size(0), -1), tea_S.view(tea_S.size(0), -1)).mean()
    positive2 = Similarity(stu_T.view(stu_T.size(0), -1), tea_T.view(tea_T.size(0), -1)).mean()
    negative1 = Similarity(stu_S.view(stu_S.size(0), -1), stu_T.view(stu_T.size(0), -1)).mean()
    negative2 = Similarity(tea_S.view(tea_S.size(0), -1), tea_T.view(tea_T.size(0), -1)).mean()
    contrast = - torch.log((positive1 + positive2) / (positive1 + positive2 + negative1 + negative2)*tao)

    return contrast


def style_contrast(stu_S, stu_T, tea_S, tea_T, tao=1):

    positive3 = Similarity(stu_T.view(stu_T.size(0), -1), tea_S.view(tea_S.size(0), -1)).mean()

    negative3 = Similarity(stu_S.view(stu_S.size(0), -1), tea_T.view(tea_T.size(0), -1)).mean()

    contrast = - torch.log((positive3) / (positive3 + negative3) * tao)
    return contrast


def transwarp_contrast(latent_stu_S, latent_stu_T, latent_tea_S, latent_tea_T,
                       gram_stu_S, gram_stu_T, gram_tea_S, gram_tea_T, tao=1):

    positive1 = Similarity(latent_stu_S.view(latent_stu_S.size(0), -1), latent_tea_S.view(latent_tea_S.size(0), -1)).mean()
    positive2 = Similarity(latent_stu_T.view(latent_stu_T.size(0), -1), latent_tea_T.view(latent_tea_T.size(0), -1)).mean()
    negative1 = Similarity(latent_stu_S.view(latent_stu_S.size(0), -1), latent_stu_T.view(latent_stu_T.size(0), -1)).mean()
    negative2 = Similarity(latent_tea_S.view(latent_tea_S.size(0), -1), latent_tea_T.view(latent_tea_T.size(0), -1)).mean()

    positive3 = Similarity(gram_stu_S.view(gram_stu_S.size(0), -1), gram_tea_S.view(gram_tea_S.size(0), -1)).mean()
    positive4 = Similarity(gram_stu_T.view(gram_stu_T.size(0), -1), gram_tea_T.view(gram_tea_T.size(0), -1)).mean()
    positive5 = Similarity(gram_stu_S.view(gram_stu_S.size(0), -1), gram_stu_T.view(gram_stu_T.size(0), -1)).mean()
    positive6 = Similarity(gram_tea_S.view(gram_tea_S.size(0), -1), gram_tea_T.view(gram_tea_T.size(0), -1)).mean()

    # train_33_06_ab4
    contrast = -torch.log((torch.exp(positive1) + torch.exp(positive2) + torch.exp(positive3) +
                           torch.exp(positive4) + torch.exp(positive5) + torch.exp(positive6)) /
                          (torch.exp(positive1) + torch.exp(positive2) + torch.exp(positive3) +
                           torch.exp(positive4) + torch.exp(positive5) + torch.exp(positive6) +
                           torch.exp(negative1) + torch.exp(negative2)) * tao).mean()

    return contrast, positive1, positive2, negative1, negative2, positive3, positive4, positive5, positive6