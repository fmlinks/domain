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

    positive5 = Similarity(gram_stu_S.view(gram_stu_S.size(0), -1), gram_stu_T.view(gram_stu_T.size(0), -1)).mean()
    positive6 = Similarity(gram_tea_S.view(gram_tea_S.size(0), -1), gram_tea_T.view(gram_tea_T.size(0), -1)).mean()

    contrast = -torch.log((torch.exp(positive1) + torch.exp(positive2) + torch.exp(positive5) + torch.exp(positive6)) /
                          (torch.exp(positive1) + torch.exp(positive2) + torch.exp(positive5) + torch.exp(positive6) +
                           torch.exp(negative1) + torch.exp(negative2)) * tao).mean()

    positive3 = positive1 # 
    positive4 = positive2 #

    return contrast, positive1, positive2, negative1, negative2, positive3, positive4, positive5, positive6


def transwarp_contrast_v2(latent_stu_S, latent_stu_T, latent_tea_S, latent_tea_T,
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
