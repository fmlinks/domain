import wandb
import numpy as np
from tqdm import tqdm
import nibabel as nib
import torch
import json
from monai.losses import DiceCELoss, DiceLoss, Dice
from monai.transforms import AsDiscrete

from configs.config import *
from data.dataset import train_loader, val_loader
from uda.UDA import TSNetTrainer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validation(val_loader, global_step):
    experiment_name = 'train03_03'
    temp_save_path = 'D:/B/data/DomainGeneralization/data/pred/'+experiment_name+'/step{}/'.format(global_step)

    os.makedirs('D:/B/data/DomainAdaptation/data/pred/'+experiment_name+'/', exist_ok=True)
    os.makedirs(temp_save_path, exist_ok=True)
    os.makedirs(temp_save_path + 'pred_image/', exist_ok=True)
    os.makedirs(temp_save_path + 'pred_image_aug/', exist_ok=True)
    os.makedirs(temp_save_path + 'latent_stu_S/', exist_ok=True)
    os.makedirs(temp_save_path + 'latent_tea_T/', exist_ok=True)
    os.makedirs(temp_save_path + 'latent_stu_T/', exist_ok=True)
    os.makedirs(temp_save_path + 'latent_tea_S/', exist_ok=True)

    model.eval()
    eval_metric = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)   # monai.losses.DiceCELoss

    with open(data_dir + split_JSON, 'r') as j:
        contents = json.loads(j.read())
        save_name_list = contents["validation"]

    with torch.no_grad():

        v_loss_supervised_stu_S = []
        v_loss_supervised_stu_T = []
        v_loss_supervised_tea_T = []
        v_loss_supervised_tea_S = []
        v_loss_supervised = []
        v_loss_contrastive = []
        v_loss_positive1 = []
        v_loss_positive2 = []
        v_loss_negative1 = []
        v_loss_negative2 = []
        v_loss_negative3 = []
        v_loss_negative4 = []
        v_loss = []

        for i, batch in enumerate(val_loader):
            # x, y = (batch["image_RA"].to(device), batch["label_RA"].to(device))
            x = batch["image"].cuda()
            y = torch.round(batch["label"]).cuda()
            x2 = batch["image_aug"].cuda()
            y2 = batch["label_aug"].cuda()
            # m2r = batch["MRAtoRA"].cuda()
            # r2m = batch["RAtoMRA"].cuda()

            model.set_input([x, y, x2, y2])
            seg_stu_S, seg_tea_T, seg_stu_T, seg_tea_S, latent_stu_S, latent_tea_T, latent_stu_T, latent_tea_S = model()


            if i < 8:
                # save start
                seg_stu_S_mid = torch.nn.Softmax(dim=1)(seg_stu_S)
                seg_stu_S_mid = seg_stu_S_mid.cpu().numpy()
                seg_stu_S_mid = np.argmax(seg_stu_S_mid, axis=1)

                npy = seg_stu_S_mid[0, :, :, :]
                new_image = nib.Nifti1Image(npy, np.eye(4), dtype=np.uint8)
                save_path = save_name_list[i]["image"].replace('/DomainGeneralization/data/image/',
                                                               '/DomainGeneralization/data/pred/' +
                                                               experiment_name +
                                                               '/step{}/pred_image/'.format(global_step))
                # print(save_path)
                if save_path is not save_name_list[i]["image"]:
                    nib.save(new_image, save_path)

                seg_tea_T_mid = torch.nn.Softmax(dim=1)(seg_tea_T)
                seg_tea_T_mid = seg_tea_T_mid.cpu().numpy()
                seg_tea_T_mid = np.argmax(seg_tea_T_mid, axis=1)

                npy = seg_tea_T_mid[0, :, :, :]
                new_image = nib.Nifti1Image(npy, np.eye(4), dtype=np.uint8)
                save_path = save_name_list[i]["image_aug"].replace('/DomainGeneralization/data/image/',
                                                                   '/DomainGeneralization/data/pred/' +
                                                                   experiment_name +
                                                                   '/step{}/pred_image_aug/'.format(global_step))
                # print(save_path)
                if save_path is not save_name_list[i]["image_aug"]:
                    nib.save(new_image, save_path)


            loss = model.get_loss()

            v_loss_supervised_stu_S.append(loss[0])
            v_loss_supervised_stu_T.append(loss[1])
            v_loss_supervised_tea_T.append(loss[2])
            v_loss_supervised_tea_S.append(loss[3])
            v_loss_supervised.append(loss[4])
            v_loss_contrastive.append(loss[5])
            v_loss_positive1.append(loss[6])
            v_loss_positive2.append(loss[7])
            v_loss_negative1.append(loss[8])
            v_loss_negative2.append(loss[9])
            v_loss_negative3.append(loss[10])
            v_loss_negative4.append(loss[11])
            v_loss.append(loss[12])

        v_loss_supervised_stu_S_avg = np.mean(v_loss_supervised_stu_S, axis=0)
        v_loss_supervised_stu_T_avg = np.mean(v_loss_supervised_stu_T, axis=0)
        v_loss_supervised_tea_T_avg = np.mean(v_loss_supervised_tea_T, axis=0)
        v_loss_supervised_tea_S_avg = np.mean(v_loss_supervised_tea_S, axis=0)
        v_loss_supervised_avg = np.mean(v_loss_supervised, axis=0)
        v_loss_contrastive_avg = np.mean(v_loss_contrastive, axis=0)
        v_loss_positive1_avg = np.mean(v_loss_positive1, axis=0)
        v_loss_positive2_avg = np.mean(v_loss_positive2, axis=0)
        v_loss_negative1_avg = np.mean(v_loss_negative1, axis=0)
        v_loss_negative2_avg = np.mean(v_loss_negative2, axis=0)
        v_loss_negative3_avg = np.mean(v_loss_negative3, axis=0)
        v_loss_negative4_avg = np.mean(v_loss_negative4, axis=0)
        v_loss_avg = np.mean(v_loss, axis=0)

        print('Loss: v_loss_supervised_stu_S_avg', v_loss_supervised_stu_S_avg)
        print('Loss: v_loss_supervised_stu_T_avg', v_loss_supervised_stu_T_avg)
        print('Loss: v_loss_supervised_tea_T_avg', v_loss_supervised_tea_T_avg)
        print('Loss: v_loss_supervised_tea_S_avg', v_loss_supervised_tea_S_avg)
        print('Loss: v_loss_supervised_avg', v_loss_supervised_avg)
        print('Loss: v_loss_positive1_avg', v_loss_positive1_avg)
        print('Loss: v_loss_positive2_avg', v_loss_positive2_avg)
        print('Loss: v_loss_negative1_avg', v_loss_negative1_avg)
        print('Loss: v_loss_negative2_avg', v_loss_negative2_avg)
        print('Loss: v_loss_negative3_avg', v_loss_negative3_avg)
        print('Loss: v_loss_negative4_avg', v_loss_negative4_avg)
        print('Loss: v_loss_contrastive_avg', v_loss_contrastive_avg)
        print('Loss: v_loss_avg', v_loss_avg)

        if wandb_record:
            wandb.log({'loss': v_loss_avg,
                       'loss_supervised_stu_S': v_loss_supervised_stu_S_avg,
                       'loss_supervised_stu_T': v_loss_supervised_stu_T_avg,
                       'loss_supervised_tea_T': v_loss_supervised_tea_T_avg,
                       'loss_supervised_tea_S': v_loss_supervised_tea_S_avg,
                       'loss_supervised': v_loss_supervised_avg,
                       'loss_contrastive': v_loss_contrastive_avg,
                       'loss_positive1': v_loss_positive1_avg,
                       'loss_positive2': v_loss_positive2_avg,
                       'loss_negative1': v_loss_negative1_avg,
                       'loss_negative2': v_loss_negative2_avg,
                       'loss_negative3': v_loss_negative3_avg,
                       'loss_negative4': v_loss_negative4_avg,
                       })

    return v_loss_avg, v_loss_supervised_avg, v_loss_contrastive_avg


def train(model, global_step, train_loader, dice_val_best, global_step_best):

    model.train()
    epoch_iterator = tqdm(train_loader, desc="aaa Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):
        # torch.cuda.empty_cache()
        step += 1
        # x, y = (batch["image_RA"].cuda(), batch["label_RA"].cuda())
        # x, y, x2 = (batch["image_RA"].cuda(), batch["label_RA"].cuda(), batch["image_MRA"].cuda())

        # x = batch[0]["image"].cuda()
        # y = torch.round(batch[0]["label"]).cuda()
        # x2 = batch[0]["image_aug"].cuda()
        # y2 = batch[0]["label_aug"].cuda()
        x = batch["image"].cuda()
        y = torch.round(batch["label"]).cuda()
        x2 = batch["image_aug"].cuda()
        y2 = batch["label_aug"].cuda()

        model.set_input([x, y, x2, y2])
        model.optimize_parameters()
        loss = model.get_loss()

        epoch_iterator.set_description \
            ("Training (%d / %d Steps)(v_loss=%2.5f)" % (global_step, max_iterations, loss[-1]))

        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = val_loader
            v_loss_avg, v_loss_supervised_avg, v_loss_contrastive_avg = validation(epoch_iterator_val, global_step)

            if v_loss_avg < dice_val_best:
                dice_val_best = v_loss_avg
                global_step_best = global_step
                torch.save(model.state_dict(),
                           os.path.join(root_dir,
                                        "train03_03-Step{}_Loss{}_DCELoss{}_ConLoss{}.pth".format(
                                            global_step,
                                            np.round(v_loss_avg, 4),
                                            np.round(v_loss_supervised_avg, 4),
                                            np.round(v_loss_contrastive_avg, 4),
                                        )))
                print("Model Saved ! Best Avg. Loss: {} Current Avg. Loss: {}".format(dice_val_best, v_loss_avg))
            else:
                print("Model Not Saved ! Best Avg. Loss: {} Current Avg. Loss: {}".format(dice_val_best, v_loss_avg))

        global_step += 1

    return global_step, dice_val_best, global_step_best


if __name__ == '__main__':

    wandb_record = False

    if wandb_record:
        # start a new wandb run to track this script
        wandb.init(
            settings=wandb.Settings(start_method='spawn'),
            # set the wandb project where this run will be logged
            project="domain generalization",

            # track hyperparameters and run metadata
            config={
                "learning_rate": 0.0003,
                "architecture": "SwinUNet",
                "dataset": "Domain generalization",
                "epochs": 40,
            }
        )

    model = TSNetTrainer().to(device)
    # model.load_state_dict(torch.load(os.path.join(root_dir, ".pth")))

    torch.backends.cudnn.benchmark = True

    global_step = 0
    dice_val_best = 10  # if Dice not DiceLoss, set 0
    global_step_best = 0
    post_label = AsDiscrete(to_onehot=2)
    post_pred = AsDiscrete(argmax=True, to_onehot=2)

    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(model, global_step, train_loader,
                                                             dice_val_best, global_step_best)
    if wandb_record:
        wandb.finish()
    # model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
