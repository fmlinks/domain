import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import wandb
import numpy as np
from tqdm import tqdm
import nibabel as nib
import torch
import json
from monai.losses import DiceCELoss, DiceLoss, Dice
from monai.transforms import AsDiscrete

from configs.config import *
from data.dataset import train_loader, val_loader  # dataset里也有版本设置
from uda.UDA import TSNetTrainer
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validation(val_loader, global_step):
    experiment_name = 'train33_06'
    # train15_03 : no diffusion
    temp_save_path = 'D:/B/data/DomainAdaptation/data/pred/'+experiment_name+'/step{}/'.format(global_step)

    os.makedirs('D:/B/data/DomainAdaptation/data/pred/'+experiment_name+'/', exist_ok=True)
    os.makedirs(temp_save_path, exist_ok=True)
    os.makedirs(temp_save_path + 'pred_3DRA/', exist_ok=True)
    os.makedirs(temp_save_path + 'pred_3DRAFDA/', exist_ok=True)
    os.makedirs(temp_save_path + 'pred_MRA/', exist_ok=True)
    os.makedirs(temp_save_path + 'pred_MRAFDA/', exist_ok=True)
    os.makedirs(temp_save_path + 'latent_stu_S/', exist_ok=True)
    os.makedirs(temp_save_path + 'latent_tea_T/', exist_ok=True)
    os.makedirs(temp_save_path + 'latent_stu_T/', exist_ok=True)
    os.makedirs(temp_save_path + 'latent_tea_S/', exist_ok=True)

    model.eval()
    eval_metric = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)   # monai.losses.DiceCELoss

    # eval_metric1 = TPDiceScore(to_onehot_y=True, softmax=True, include_background=False, lossdim=[1])
    # eval_metric2 = TPDiceScore(to_onehot_y=True, softmax=True, include_background=False, lossdim=[2])
    # eval_metric3 = MRATPDiceScore(to_onehot_y=True, softmax=True, include_background=False, lossdim=[1])

    # eval_metric1 = dice_coefficient_3d_ra_vessel
    # eval_metric2 = dice_coefficient_3d_ra_aneurysm
    # eval_metric3 = dice_coefficient_3d_mra_aneurysm

    # eval_metric_mra = MRADiceScore(weight=[0.001, 0.999])

    with open(data_dir + split_JSON, 'r') as j:
        contents = json.loads(j.read())
        # print(len(contents["validation"]))
        save_name_list = contents["validation"]
        # print(data_dir + save_name_list[0]["image_RA"].replace('./imagesTr/', './pred_3DRA/'))
        # print(data_dir + save_name_list[0]["image_MRA"].replace('./imagesTrMRA/', './pred_MRA/'))
        # print(data_dir + save_name_list[0]["MRAtoRA"].replace('./imagesTrMRAFDA/', './pred_MRAFDA/'))
        # print(data_dir + save_name_list[0]["RAtoMRA"].replace('./imagesTrFDA/', './pred_3DRAFDA/'))

    with torch.no_grad():
        metric1_2 = []
        metric2_2 = []
        metric3_2 = []
        metric4_2 = []
        metric5_2 = []
        dice_ra_an_all = []
        dice_ra_vt_all = []
        dice_mra_an_all = []
        dice_mra_an_seg_stu_T_all = []

        v_loss_supervised_loss_student_all = []
        v_loss_supervised_loss_student_MRA_all = []
        v_loss_supervised_all = []
        v_loss_semi_mse_loss_source_all = []
        v_loss_semi_mse_loss_target_all = []
        v_loss_semi_sim_loss_source_all = []
        v_loss_semi_sim_loss_target_all = []
        v_loss_semi_all = []
        v_loss_contrastive_instance_all = []
        v_loss_contrastive_style_all = []
        v_loss_contrastive_transwarp_all = []
        v_loss_positive1_all = []
        v_loss_positive2_all = []
        v_loss_negative1_all = []
        v_loss_negative2_all = []
        v_loss_positive3_all = []
        v_loss_positive4_all = []
        v_loss_positive5_all = []
        v_loss_positive6_all = []

        v_loss_all = []

        for i, batch in enumerate(val_loader):
            # x, y = (batch["image_RA"].to(device), batch["label_RA"].to(device))
            x = batch[0]["image_RA"].cuda()
            y = torch.round(batch[0]["label_RA"]).cuda()
            x2 = batch[0]["image_MRA"].cuda()
            y2 = batch[0]["label_MRA"].cuda()
            # m2r = batch["MRAtoRA"].cuda()
            # r2m = batch["RAtoMRA"].cuda()

            # model.set_input([x, y, x2, m2r, r2m])
            model.set_input([x, y, x2, y2])
            seg_stu_S, seg_tea_T, seg_stu_T, seg_tea_S, \
                latent_stu_S, latent_tea_T, latent_stu_T, latent_tea_S, \
                gram_stu_S, gram_tea_T, gram_stu_T, gram_tea_S = model()



            # save start
            seg_stu_S_mid = torch.nn.Softmax(dim=1)(seg_stu_S)
            seg_stu_S_mid = seg_stu_S_mid.cpu().numpy()
            seg_stu_S_mid = np.argmax(seg_stu_S_mid, axis=1)

            npy = seg_stu_S_mid[0, :, :, :]
            new_image = nib.Nifti1Image(npy, np.eye(4), dtype=np.uint8)
            save_path = save_name_list[i]["image_RA"].replace('/DomainAdaptation/data/val/image/',
                                                              '/DomainAdaptation/data/pred/' +
                                                              experiment_name +
                                                              '/step{}/pred_3DRA/'.format(global_step))
            # print(save_path)
            nib.save(new_image, save_path)

            seg_tea_T_mid = torch.nn.Softmax(dim=1)(seg_tea_T)
            seg_tea_T_mid = seg_tea_T_mid.cpu().numpy()
            seg_tea_T_mid = np.argmax(seg_tea_T_mid, axis=1)

            npy = seg_tea_T_mid[0, :, :, :]
            new_image = nib.Nifti1Image(npy, np.eye(4), dtype=np.uint8)
            save_path = save_name_list[i]["image_MRA"].replace('/SMILE-UHURA/data/val/image/',
                                                              '/DomainAdaptation/data/pred/' +
                                                              experiment_name +
                                                              '/step{}/pred_MRA/'.format(global_step))
            # print(save_path)
            nib.save(new_image, save_path)


            # latent_stu_S
            data_to_save = {'latent_stu_S': latent_stu_S.cpu().numpy(),
                            'gram_stu_S': gram_stu_S.cpu().numpy()}
            save_path = save_name_list[i]["image_RA"].replace('/DomainAdaptation/data/val/image/',
                                                              '/DomainAdaptation/data/pred/' +
                                                              experiment_name +
                                                              '/step{}/latent_stu_S/'.format(global_step))
            save_path = save_path.replace('.nii.gz', '.pkl')
            with open(save_path, 'wb') as file:
                pickle.dump(data_to_save, file)
            print(f"Saved variables to {save_path}")

            # latent_tea_T
            data_to_save = {'latent_tea_T': latent_tea_T.cpu().numpy(),
                            'gram_tea_T': gram_tea_T.cpu().numpy()}
            save_path = save_name_list[i]["image_MRA"].replace('/SMILE-UHURA/data/val/image/',
                                                              '/DomainAdaptation/data/pred/' +
                                                              experiment_name +
                                                              '/step{}/latent_tea_T/'.format(global_step))
            save_path = save_path.replace('.nii.gz', '.pkl')
            with open(save_path, 'wb') as file:
                pickle.dump(data_to_save, file)
            print(f"Saved variables to {save_path}")

            # latent_stu_T
            data_to_save = {'latent_stu_T': latent_stu_T.cpu().numpy(),
                            'gram_stu_T': gram_stu_T.cpu().numpy()}
            save_path = save_name_list[i]["image_MRA"].replace('/SMILE-UHURA/data/val/image/',
                                                                '/DomainAdaptation/data/pred/' +
                                                                experiment_name +
                                                                '/step{}/latent_stu_T/'.format(global_step))
            save_path = save_path.replace('.nii.gz', '.pkl')
            with open(save_path, 'wb') as file:
                pickle.dump(data_to_save, file)
            print(f"Saved variables to {save_path}")

            # latent_tea_S
            data_to_save = {'latent_tea_S': latent_tea_S.cpu().numpy(),
                            'gram_tea_S': gram_tea_S.cpu().numpy()}
            save_path = save_name_list[i]["image_RA"].replace('/DomainAdaptation/data/val/image/',
                                                                '/DomainAdaptation/data/pred/' +
                                                                experiment_name +
                                                                '/step{}/latent_tea_S/'.format(global_step))
            save_path = save_path.replace('.nii.gz', '.pkl')
            with open(save_path, 'wb') as file:
                pickle.dump(data_to_save, file)
            print(f"Saved variables to {save_path}")
            # save end






            # seg_stu_T_mid = torch.nn.Softmax(dim=1)(seg_stu_T)
            # seg_stu_T_mid = seg_stu_T_mid.cpu().numpy()
            # seg_stu_T_mid = np.argmax(seg_stu_T_mid, axis=1)
            #
            # npy = seg_stu_T_mid[0, :, :, :]
            # new_image = nib.Nifti1Image(npy, np.eye(4), dtype=np.uint8)
            # save_path = save_name_list[i]["image_MRA"].replace('/SMILE-UHURA/patch256/val/image/',
            #                                                   '/DomainAdaptation/patch256/pred/' +
            #                                                   experiment_name +
            #                                                   '/step{}/pred_MRAFDA/'.format(global_step))
            # # print(save_path)
            # nib.save(new_image, save_path)

            # npy = seg_stu_S[0, 0, :, :, :].cpu().numpy()
            # new_image = nib.Nifti1Image(npy, np.eye(4))
            # new_image.set_data_dtype(np.uint8)
            # save_path = data_dir + save_name_list[i]["image_RA"].replace('./imagesTr/', './pred_3DRA/channel0/')
            # print(save_path)
            # # nib.save(new_image, save_path)
            #
            # npy = seg_stu_S[0, 1, :, :, :].cpu().numpy()
            # new_image = nib.Nifti1Image(npy, np.eye(4))
            # new_image.set_data_dtype(np.uint8)
            # save_path = data_dir + save_name_list[i]["image_RA"].replace('./imagesTr/', './pred_3DRA/channel1/')
            # nib.save(new_image, save_path)
            #
            # npy = seg_stu_S[0, 2, :, :, :].cpu().numpy()
            # new_image = nib.Nifti1Image(npy, np.eye(4))
            # new_image.set_data_dtype(np.uint8)
            # save_path = data_dir + save_name_list[i]["image_RA"].replace('./imagesTr/', './pred_3DRA/channel2/')
            # nib.save(new_image, save_path)

            loss = model.get_loss()

            v_loss_supervised_loss_student_all.append(loss[0])
            v_loss_supervised_loss_student_MRA_all.append(loss[1])
            v_loss_supervised_all.append(loss[2])
            v_loss_semi_mse_loss_source_all.append(loss[3])
            v_loss_semi_mse_loss_target_all.append(loss[4])
            v_loss_semi_sim_loss_source_all.append(loss[5])
            v_loss_semi_sim_loss_target_all.append(loss[6])
            v_loss_semi_all.append(loss[7])
            v_loss_contrastive_instance_all.append(loss[8])
            v_loss_contrastive_style_all.append(loss[9])
            v_loss_contrastive_transwarp_all.append(loss[10])
            v_loss_positive1_all.append(loss[11])
            v_loss_positive2_all.append(loss[12])
            v_loss_negative1_all.append(loss[13])
            v_loss_negative2_all.append(loss[14])
            v_loss_positive3_all.append(loss[15])
            v_loss_positive4_all.append(loss[16])
            v_loss_positive5_all.append(loss[17])
            v_loss_positive6_all.append(loss[18])
            v_loss_all.append(loss[-1])

            # dice_ra_an = dice_coefficient_3d_ra_aneurysm(seg_stu_S, y).item()
            # dice_ra_vt = dice_coefficient_3d_ra_vessel(seg_stu_S, y).item()
            # dice_mra_an = dice_coefficient_3d_mra_aneurysm(seg_tea_T, y).item()
            # dice_mra_an_seg_stu_T = dice_coefficient_3d_mra_aneurysm(seg_stu_T, y).item()
            #
            # dice_ra_an_all.append(dice_ra_an)
            # dice_ra_vt_all.append(dice_ra_vt)
            # dice_mra_an_all.append(dice_mra_an)
            # dice_mra_an_seg_stu_T_all.append(dice_mra_an_seg_stu_T)
            #
            # seg_tea_T = np.delete(seg_tea_T, 1, axis=1)
            # seg_tea_T = torch.from_numpy(seg_tea_T).to(device)
            # metric1_0 = eval_metric_mra(seg_tea_T, y2)
            # metric1_1 = metric1_0.item()
            # metric1_2.append(metric1_1)
            #
            # seg_stu_T = np.delete(seg_stu_T, 1, axis=1)
            # seg_stu_T = torch.from_numpy(seg_stu_T).to(device)
            # metric3_0 = eval_metric_mra(seg_stu_T, y2)
            # metric3_1 = metric3_0.item()
            # metric3_2.append(metric3_1)
            #
            metric2_0 = eval_metric(seg_stu_S, y)
            metric2_1 = metric2_0.item()
            metric2_2.append(metric2_1)
            #
            metric3_0 = eval_metric(seg_tea_T, y2)
            metric3_1 = metric3_0.item()
            metric3_2.append(metric3_1)
            #
            metric4_0 = eval_metric(seg_stu_T, y2)
            metric4_1 = metric4_0.item()
            metric4_2.append(metric4_1)

            metric5_0 = eval_metric(seg_tea_S, y)
            metric5_1 = metric5_0.item()
            metric5_2.append(metric5_1)

        metric2_3 = np.mean(metric2_2, axis=0)
        print('\n')
        print('1 - DiceLoss no Background')
        print('3DRA Vessel Dice: ', 1-metric2_3)
        metric3_3 = np.mean(metric3_2, axis=0)
        print('MRA Vessel Dice: ', 1-metric3_3)
        metric4_3 = np.mean(metric4_2, axis=0)
        print('MRA to 3DRA Vessel Dice: ', 1-metric4_3)
        metric5_3 = np.mean(metric5_2, axis=0)
        print('3DRA to MRA Vessel Dice: ', 1-metric5_3)

        # metric1_3 = np.mean(metric1_2, axis=0)
        # print('DiceScore: MRA Aneurysm teacher: ', metric1_3)
        # metric3_3 = np.mean(metric3_2, axis=0)
        # print('DiceScore: MRA Aneurysm seg_stu_T: ', metric3_3)
        #
        # dice_ra_an_avg = np.mean(dice_ra_an_all, axis=0)
        # dice_ra_vt_avg = np.mean(dice_ra_vt_all, axis=0)
        # dice_mra_an_avg = np.mean(dice_mra_an_all, axis=0)
        # dice_mra_an_seg_stu_T_avg = np.mean(dice_mra_an_seg_stu_T_all, axis=0)
        #
        # print('DSC: 3DRA Aneurysm: ', dice_ra_an_avg)
        # print('DSC: 3DRA Vessel: ', dice_ra_vt_avg)
        # print('DSC: MRA Aneurysm: ', dice_mra_an_avg)
        # print('DSC: MRA Aneurysm seg_stu_T: ', dice_mra_an_seg_stu_T_avg)
        # print('DSC: 3DRA', dice_ra_an_avg * 0.8 + dice_ra_vt_avg * 0.2)

        v_loss_supervised_loss_student_avg = np.mean(v_loss_supervised_loss_student_all, axis=0)
        v_loss_supervised_loss_student_MRA_avg = np.mean(v_loss_supervised_loss_student_MRA_all, axis=0)
        v_loss_supervised_avg = np.mean(v_loss_supervised_all, axis=0)
        v_loss_semi_mse_loss_source_avg = np.mean(v_loss_semi_mse_loss_source_all, axis=0)
        v_loss_semi_mse_loss_target_avg = np.mean(v_loss_semi_mse_loss_target_all, axis=0)
        v_loss_semi_sim_loss_source_avg = np.mean(v_loss_semi_sim_loss_source_all, axis=0)
        v_loss_semi_sim_loss_target_avg = np.mean(v_loss_semi_sim_loss_target_all, axis=0)
        v_loss_semi_avg = np.mean(v_loss_semi_all, axis=0)
        # v_loss_contrastive_instance_avg = np.mean(v_loss_contrastive_instance_all, axis=0)
        # v_loss_contrastive_style_avg = np.mean(v_loss_contrastive_style_all, axis=0)
        v_loss_contrastive_instance_avg = 0
        v_loss_contrastive_style_avg = 0
        v_loss_contrastive_transwarp_avg = np.mean(v_loss_contrastive_transwarp_all, axis=0)
        v_loss_positive1_avg = np.mean(v_loss_positive1_all, axis=0)
        v_loss_positive2_avg = np.mean(v_loss_positive2_all, axis=0)
        v_loss_negative1_avg = np.mean(v_loss_negative1_all, axis=0)
        v_loss_negative2_avg = np.mean(v_loss_negative2_all, axis=0)
        v_loss_positive3_avg = np.mean(v_loss_positive3_all, axis=0)
        v_loss_positive4_avg = np.mean(v_loss_positive4_all, axis=0)
        v_loss_positive5_avg = np.mean(v_loss_positive5_all, axis=0)
        v_loss_positive6_avg = np.mean(v_loss_positive6_all, axis=0)
        v_loss_avg = np.mean(v_loss_all, axis=0)

        print('Loss: v_loss_supervised_loss_student_avg', v_loss_supervised_loss_student_avg)
        print('Loss: v_loss_supervised_loss_student_MRA_avg', v_loss_supervised_loss_student_MRA_avg)
        print('Loss: v_loss_supervised_avg', v_loss_supervised_avg)
        print('Loss: v_loss_semi_mse_loss_source_avg', v_loss_semi_mse_loss_source_avg)
        print('Loss: v_loss_semi_mse_loss_target_avg', v_loss_semi_mse_loss_target_avg)
        print('Loss: v_loss_semi_sim_loss_source_avg', v_loss_semi_sim_loss_source_avg)
        print('Loss: v_loss_semi_sim_loss_target_avg', v_loss_semi_sim_loss_target_avg)
        print('Loss: v_loss_semi_avg', v_loss_semi_avg)
        print('Loss: v_loss_contrastive_instance_avg', v_loss_contrastive_instance_avg)
        print('Loss: v_loss_contrastive_style_avg', v_loss_contrastive_style_avg)
        print('Loss: v_loss_contrastive_transwarp_avg', v_loss_contrastive_transwarp_avg)
        print('Loss: v_loss_avg', v_loss_avg)
        print('Loss: v_loss_positive1_avg', v_loss_positive1_avg)
        print('Loss: v_loss_positive2_avg', v_loss_positive2_avg)
        print('Loss: v_loss_negative1_avg', v_loss_negative1_avg)
        print('Loss: v_loss_negative2_avg', v_loss_negative2_avg)
        print('Loss: v_loss_positive3_avg', v_loss_positive3_avg)
        print('Loss: v_loss_positive4_avg', v_loss_positive4_avg)
        print('Loss: v_loss_positive5_avg', v_loss_positive5_avg)
        print('Loss: v_loss_positive6_avg', v_loss_positive6_avg)



        if wandb_record:
            wandb.log({'loss': v_loss_avg,
                       'loss_supervised_loss_student': v_loss_supervised_loss_student_avg,
                       'loss_supervised_loss_student_MRA': v_loss_supervised_loss_student_MRA_avg,
                       'loss_supervised': v_loss_supervised_avg,
                       'loss_semi_mse_loss_source': v_loss_semi_mse_loss_source_avg,
                       'loss_semi_mse_loss_target': v_loss_semi_mse_loss_target_avg,
                       'loss_semi_sim_loss_source': v_loss_semi_sim_loss_source_avg,
                       'loss_semi_sim_loss_target': v_loss_semi_sim_loss_target_avg,
                       'loss_semi': v_loss_semi_avg,
                       'loss_contrastive_instance': v_loss_contrastive_instance_avg,
                       'loss_contrastive_style': v_loss_contrastive_style_avg,
                       'loss_contrastive_transwarp': v_loss_contrastive_transwarp_avg,
                       'loss_positive1-latent_stu_S-latent_tea_S': v_loss_positive1_avg,
                       'loss_positive2-latent_stu_T-latent_tea_T': v_loss_positive2_avg,
                       'loss_negative1-latent_stu_S-latent_stu_T': v_loss_negative1_avg,
                       'loss_negative2-latent_tea_S-latent_tea_T': v_loss_negative2_avg,
                       'loss_positive3-gram_stu_S-gram_tea_S': v_loss_positive3_avg,
                       'loss_positive4-gram_stu_T-gram_tea_T': v_loss_positive4_avg,
                       'loss_positive5-gram_stu_T-gram_tea_S': v_loss_positive5_avg,
                       'loss_positive6-gram_stu_S-gram_tea_T': v_loss_positive6_avg,
                       '3DRA Vessel Dice': 1 - metric2_3,
                       'MRA Vessel Dice': 1 - metric3_3,
                       'MRA to 3DRA Vessel Dice': 1 - metric4_3,
                       '3DRA to MRA Vessel Dice': 1 - metric5_3})

    return v_loss_avg, v_loss_supervised_loss_student_avg, v_loss_supervised_loss_student_MRA_avg, 1-metric3_3


def train(model, global_step, train_loader, dice_val_best, global_step_best):

    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="aaa Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):
        # torch.cuda.empty_cache()
        step += 1
        # x, y = (batch["image_RA"].cuda(), batch["label_RA"].cuda())
        # x, y, x2 = (batch["image_RA"].cuda(), batch["label_RA"].cuda(), batch["image_MRA"].cuda())
        batch = batch
        x = batch[0]["image_RA"].cuda()
        y = torch.round(batch[0]["label_RA"]).cuda()
        x2 = batch[0]["image_MRA"].cuda()
        y2 = batch[0]["pseudo_MRA"].cuda()
        # x = batch["image_RA"].cuda()
        # y = torch.round(batch["label_RA"]).cuda()
        # x2 = batch["image_MRA"].cuda()
        # y2 = batch["pseudo_MRA"].cuda()



        # y2 = batch[0]["label_MRA"].cuda()
        # print(x.shape, y.shape, x2.shape, 'aaaaaaaaaaaaaaaa')

        # m2r = batch["MRAtoRA"].cuda()
        # r2m = batch["RAtoMRA"].cuda()

        # for i in range(batch["image_RA"].shape[0]):
        #     npy = batch["image_RA"][i, 0, :, :, :].numpy()
        #     new_image = nib.Nifti1Image(npy, np.eye(4))
        #     new_image.set_data_dtype(np.uint8)
        #     nib.save(new_image, '../temp/train06_val_image_{}.nii.gz'.format(i))
        #
        #     npy = batch["label_RA"][i, 0, :, :, :].numpy()
        #     new_image = nib.Nifti1Image(npy, np.eye(4))
        #     new_image.set_data_dtype(np.uint8)
        #     nib.save(new_image, '../temp/train06_val_label_{}.nii.gz'.format(i))
        #
        #     npy = batch["image_MRA"][i, 0, :, :, :].numpy()
        #     new_image = nib.Nifti1Image(npy, np.eye(4))
        #     new_image.set_data_dtype(np.uint8)
        #     nib.save(new_image, '../temp/train06_vaal_image_MRA_{}.nii.gz'.format(i))
        #
        #     npy = batch["label_MRA"][i, 0, :, :, :].numpy()
        #     new_image = nib.Nifti1Image(npy, np.eye(4))
        #     new_image.set_data_dtype(np.uint8)
        #     nib.save(new_image, '../temp/train06_val_label_MRA_{}.nii.gz'.format(i))

        # 这里需要用函数生成6个input,顺序是RA_img, RA_label, RAtoMRA_img, RAtoMRA_label, MRA_img, MRAtoRA_img ******
        # model.set_input([x, y, x2, y2])
        model.set_input([x, y, x2, y2])
        model.optimize_parameters()
        loss = model.get_loss()

        epoch_iterator.set_description \
            ("Training (%d / %d Steps)(v_loss=%2.5f)" % (global_step, max_iterations, loss[-1]))

        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = val_loader
            # epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            v_loss_avg, v_loss_supervised_loss_student_avg, v_loss_supervised_loss_teacher_avg, dice_val \
                = validation(epoch_iterator_val, global_step)
            # dice_val = validation(epoch_iterator_val)

            # torch.save(model.state_dict(),
            #            os.path.join(root_dir,
            #                         "train21_01-Step{}_Loss{}_StuLoss{}_TeaLoss{}.pth".format(
            #                             global_step,
            #                             np.round(v_loss_avg, 4),
            #                             np.round(v_loss_supervised_loss_student_avg, 4),
            #                             np.round(v_loss_supervised_loss_teacher_avg, 4),
            #                         )))

            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(),
                           os.path.join(root_dir,
                                        "train33_06-Step{}_Loss{}_StuLoss{}_TeaLoss{}.pth".format(
                                            global_step,
                                            np.round(v_loss_avg, 4),
                                            np.round(v_loss_supervised_loss_student_avg, 4),
                                            np.round(v_loss_supervised_loss_teacher_avg, 4),
                                        )))
                print("Model Saved ! Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val))
            else:
                print("Model Not Saved ! Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val))

        global_step += 1

    return global_step, dice_val_best, global_step_best


if __name__ == '__main__':

    wandb_record = False

    if wandb_record:
        # start a new wandb run to track this script
        wandb.init(
            settings=wandb.Settings(start_method='spawn'),
            # set the wandb project where this run will be logged
            project="domain adaptation",

            # track hyperparameters and run metadata
            config={
                "learning_rate": 0.0001,
                "architecture": "SwinUNet",
                "dataset": "Domain Adaptation",
                "epochs": 100,
            }
        )

    model = TSNetTrainer().to(device)
    model.load_state_dict(torch.load(os.path.join(root_dir,"train33_06-Step208_Loss0.2316_StuLoss0.3559_TeaLoss0.1644.pth")))

    torch.backends.cudnn.benchmark = True
    # loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    global_step = 0
    dice_val_best = 0  # if Dice not DiceLoss, set 0
    global_step_best = 0
    post_label = AsDiscrete(to_onehot=2)
    post_pred = AsDiscrete(argmax=True, to_onehot=2)

    # train_loader, val_loader = get_data()
    # write a dice loss
    # loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    # loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    # loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True)

    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(model, global_step, train_loader,
                                                             dice_val_best, global_step_best)
    if wandb_record:
        wandb.finish()
    # model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))