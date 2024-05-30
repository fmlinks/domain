import torch
import torch.nn as nn

Similarity = nn.CosineSimilarity(dim=-1)

cosine_loss = nn.CosineEmbeddingLoss(margin=0.5)

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

    contrast = -torch.log((torch.exp(positive1) + torch.exp(positive2) + torch.exp(positive3) +
                           torch.exp(positive4) + torch.exp(positive5) + torch.exp(positive6)) /
                          (torch.exp(positive1) + torch.exp(positive2) + torch.exp(positive3) +
                           torch.exp(positive4) + torch.exp(positive5) + torch.exp(positive6) +
                           torch.exp(negative1) + torch.exp(negative2)) * tao).mean()

    return contrast, positive1, positive2, negative1, negative2, positive3, positive4, positive5, positive6


def transwarp_contrast_v1(latent_stu_S, latent_stu_T, latent_tea_S, latent_tea_T,
                       gram_stu_S, gram_stu_T, gram_tea_S, gram_tea_T, tao=1):

    cosine_loss = nn.CosineEmbeddingLoss(margin=0.5)

    positive1 = latent_stu_S.view(latent_stu_S.size(0), -1)
    positive1_target = latent_tea_S.view(latent_tea_S.size(0), -1)
    positive2 = latent_stu_T.view(latent_stu_T.size(0), -1)
    positive2_target = latent_tea_T.view(latent_tea_T.size(0), -1)

    negative1 = latent_stu_S.view(latent_stu_S.size(0), -1)
    negative1_target = latent_stu_T.view(latent_stu_T.size(0), -1)
    negative2 = latent_tea_S.view(latent_tea_S.size(0), -1)
    negative2_target = latent_tea_T.view(latent_tea_T.size(0), -1)

    positive3 = gram_stu_T.view(gram_stu_T.size(0), -1)
    positive3_target = gram_tea_S.view(gram_tea_S.size(0), -1)
    negative3 = gram_stu_S.view(gram_stu_S.size(0), -1)
    negative3_target = gram_tea_T.view(gram_tea_T.size(0), -1)

    loss_positive1 = cosine_loss(positive1, positive1_target, torch.ones(1).to(positive1.device))
    loss_positive2 = cosine_loss(positive2, positive2_target, torch.ones(1).to(positive2.device))
    loss_negative1 = cosine_loss(negative1, negative1_target, -torch.ones(1).to(negative1.device))
    loss_negative2 = cosine_loss(negative2, negative2_target, -torch.ones(1).to(negative2.device))

    loss_positive3 = cosine_loss(positive3, positive3_target, torch.ones(1).to(positive3.device))
    loss_negative3 = cosine_loss(negative3, negative3_target, -torch.ones(1).to(negative3.device))

    contrast = loss_positive1 + loss_positive2 + loss_positive3 + loss_negative1 + loss_negative2 + loss_negative3

    return contrast, loss_positive1, loss_positive2, loss_positive3, loss_negative1, loss_negative2, loss_negative3
