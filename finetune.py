import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from typing import Tuple, Literal, Dict, Any
from dataset import CableDataset
from torch.utils.data import Dataset, DataLoader
from losses import *
import argparse


model_type = 'vit_h'
checkpoint = 'ckpts/base_model/sam_vit_h_4b8939.pth'
#dataset_train_path = 'dataset/train/train'
#dataset_test_path = 'dataset/test/test'
#dataset_train_path = '../data/cable_dataset/train/train'
#dataset_test_path = '../data/cable_dataset/test/test'
input_ori_shape = (720, 1280)

# if envionment variable: DEBUG_ON = True / true / 1
DEBUG_MODE = os.getenv('DEBUG_ON', 'False').lower() in ('true', '1')
print('Debug mode: ' + ('ON' if DEBUG_MODE is True else 'OFF'))



def update_learning_rate(
        optimizer: torch.optim.Optimizer, 
        current_iter: int, 
        warmup_iters: int, 
        initial_lr: float
    ):
    if current_iter < warmup_iters:  
        # lr linear warm up  
        lr = initial_lr * (current_iter / warmup_iters)  
        for param_group in optimizer.param_groups:  
            param_group['lr'] = lr  
    else:
        pass
        # do nothing
    



### Run fine tuning
# This is the main training loop.
# Improvements to be made include batching and moving the computation of the image and prompt embeddings outside the loop since we are not tuning these parts of the model, this will speed up training as we should not recompute the embeddings during each epoch. Sometimes the optimizer gets lost in the parameter space and the loss function blows up. Restarting from scratch (including running all cells below 'Prepare Fine Tuning' in order to start with default weights again) will solve this.
# In a production implementation a better choice of optimiser/loss function will certainly help.

from statistics import mean

from tqdm import tqdm
from torch.nn.functional import threshold, normalize

from segment_anything import SamPredictor, sam_model_registry

def finetune(args):
    # Setup

    init_lr = args.lr                               # init_lr = 1e-4
    num_epoches = args.epoch_num
    save_epoch_interval = args.save_interval
    exp_path = os.path.join('exp', args.exp_name)   #'exp/finetune_first_test'
    batch_size = args.batch_size
    device = args.device
    no_log_flag = args.no_log
    dataset_train_path = args.data_dir
    with_test_dataset = args.test_dir is not None
    if with_test_dataset:
        dataset_test_path = args.test_dir

    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'details'), exist_ok=True)

    ########################
    # Loss function choosing
    
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
    dice_loss_fn = DiceLoss()
    loss_fn = lambda x, y: 20.0 * focal_loss_fn(x, y) + 1.0 * dice_loss_fn(x, y)
    #loss_fn = None
    #if args.loss == 0:
    #    loss_fn = torch.nn.MSELoss()
    #elif args.loss == 1:
    #    loss_fn = torch.nn.BCELoss()
    ########################


    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)
    sam_model.train()

    print(f'dataset_size: {sam_model.image_encoder.img_size}')
    train_dataset = CableDataset(dataset_train_path, input_ori_shape, sam_model.image_encoder.img_size)
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True
    )

    if with_test_dataset:
        test_dataset = CableDataset(dataset_test_path, input_ori_shape, sam_model.image_encoder.img_size)
        test_dataloader = DataLoader(
            dataset=test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4, 
            pin_memory=True
        )
    

    # Set up the optimizer, hyperparameter tuning will improve performance here
    wd = 0
    warm_up_iterations = 250
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=0, weight_decay=wd)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #    optimizer=optimizer, 
    #    milestones=[6000, 8666],
    #    gamma=0.2
    #)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, 
        milestones=[600, 1200, 1800, 2400, 3000, 4000, 6000, 9000, 12000, 15000],
        gamma=0.6
    )

    def save_checkpoint(iter_num):
        tqdm.write('Saving checkpoint...')
        os.makedirs(os.path.join(exp_path, 'ckpts'), exist_ok=True)
        torch.save(sam_model.state_dict(), os.path.join(exp_path, 'ckpts', 'ckpt_{:0>5d}_b{}.pth'.format(iter_num, batch_size)))
        torch.save(optimizer.state_dict(), os.path.join(exp_path, 'ckpts', 'optimizer_{:0>5d}_b{}.pth'.format(iter_num, batch_size)))
        tqdm.write('Checkpoint saved.')

    def log_init_info():
        t_params_img_encoder = sum(p.numel() for p in sam_model.image_encoder.parameters() if p.requires_grad)
        t_params_prompt_encoder = sum(p.numel() for p in sam_model.prompt_encoder.parameters() if p.requires_grad)
        t_params_mask_decoder = sum(p.numel() for p in sam_model.mask_decoder.parameters() if p.requires_grad)
        t_params_total = sum(p.numel() for p in sam_model.parameters() if p.requires_grad) 
        log_str = f'''
Training parameters:
    learning rate (initial, after warm-up): {init_lr} 
    max epoch: {num_epoches} 
    batch_size: {batch_size} 
    checkpoint save interval (num of epoches): {save_epoch_interval} 
    device: {device} 
    scheduler: {lr_scheduler}
        milestones: {lr_scheduler.milestones}
        gamma: {lr_scheduler.gamma}
Loss funtion: {loss_fn}
Model parameters:
    image encoder: {t_params_img_encoder}
    prompt encoder: {t_params_prompt_encoder}
    mask decoder: {t_params_mask_decoder}
    total: {t_params_total}
Other information:
    do no log to wandb: {no_log_flag}
    exp name: {args.exp_name}
'''
        print(log_str)
        with open(os.path.join(exp_path, 'details', 'init_logs.txt'), 'w') as log_file:  
            log_file.write(log_str)  

    log_init_info()

    #####################################################################
    ### WandB login
    import wandb
    if not no_log_flag:
        wandb.login(key=os.getenv('WANDB_API_KEY'))

        run = wandb.init(
            # Set the project where this run will be logged
            project='SAM-finetuning', 
            name=args.exp_name, 
            # Track hyperparameters and run metadata
            config={
                "epochs": num_epoches
            }
        )


    ### training info preparation

    all_losses = []
    iteration_count = 0
    last_ckpt_saved_epoch = 0
    iter_bar = tqdm(range(num_epoches * len(train_dataset)), dynamic_ncols=True)

    def forward_loop(
            sample: Dict[Literal['image', 'mask', 'name'], Any], 
            mode: Literal['train', 'test'] = 'train', 
            cur_epoch: int = 0
        ) -> Tuple[float, np.ndarray]:
        """
        Forward training / test iteration
        Returns:
            - loss
            - (true positive, true negative, false positive, false negative), only if mode == 'test'
        """
        nonlocal iteration_count

        input_image = sample['image'].to(device)
        gt_mask = sample['mask'].to(device)
        cur_batch = input_image.shape[0]

        transformed_image = sam_model.preprocess(input_image)
        input_shape = tuple(input_image.shape[-2:])

        if mode == 'train':
            iteration_count += 1
            update_learning_rate(optimizer, iteration_count, warm_up_iterations, init_lr)

        ################=================== Forward ===================################
        # No grad here as we don't want to optimise the encoders
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(transformed_image)

            prompt_box = torch.tensor(
                [0, 0, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size], 
                dtype=torch.float, 
                device=device
            )
            prompt_box = prompt_box.tile((cur_batch, 1))

            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=prompt_box,
                masks=None,
            )
        
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_shape, input_ori_shape).to(device)
        # set negative values to 0
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
        gt_binary_mask = torch.as_tensor(gt_mask, dtype=torch.float32)

        loss = loss_fn(binary_mask, gt_binary_mask)

        #####
        if DEBUG_MODE:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(binary_mask[0].detach().cpu().numpy().squeeze())
            plt.title('predicted')
            plt.subplot(1, 3, 2)
            plt.imshow(gt_binary_mask[0].detach().cpu().numpy().squeeze())
            plt.title('g.t.')
            plt.subplot(1, 3, 3)
            plt.imshow(input_image[0].detach().cpu().numpy().squeeze().transpose(1, 2, 0))
            plt.title('original')
            plt.savefig(os.path.join(exp_path, 'details', 'masks.png'))
        ################=================== Forward ===================################

        tp, tn, fp, fn = 0, 0, 0, 0

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            iter_bar.update(cur_batch)
            iter_bar.set_description('[epoch:{}, iter:{}, loss:{:.4f}, lr:{:.6f}]'.format(cur_epoch, iteration_count, loss.item(), optimizer.param_groups[0]['lr']))

            if not no_log_flag:
                wandb.log({'training_loss': loss.item()}, step=iteration_count)
        elif mode == 'test':
            pred_bool_masks = torch.as_tensor(binary_mask, dtype=torch.bool)
            gt_bool_masks = torch.as_tensor(gt_mask, dtype=torch.bool)
            tp = torch.sum(pred_bool_masks & gt_bool_masks).item()
            tn = torch.sum(~pred_bool_masks & ~gt_bool_masks).item()
            fp = torch.sum(pred_bool_masks & ~gt_bool_masks).item()
            fn = torch.sum(~pred_bool_masks & gt_bool_masks).item()

        #plt.plot(all_losses)
        #plt.title(f'Loss curve (batch size = {train_dataloader.batch_size})')
        #plt.xlabel('Iteration')
        #plt.ylabel('Loss')
        #plt.savefig(os.path.join(exp_path, 'details', 'loss_curve.png'))
        #plt.close()
        return loss.item(), np.array([tp, tn, fp, fn])

    # end func forward_loop()

    ################# training loop #################
    for e in range(1, num_epoches + 1):
        epoch_losses = []
        
        # Iterate over the dataset
        for i, train_sample in enumerate(train_dataloader):
            cur_loss, _ = forward_loop(train_sample, mode='train', cur_epoch=e)
            epoch_losses.append(cur_loss)
            all_losses.append(cur_loss)

            ######################################## DELETE THIS!!!
            if iteration_count % 10 == 0:
                test_loss = 0
                for j, test_sample in enumerate(test_dataloader):
                    with torch.no_grad():
                        input_image = test_sample['image'].to(device)
                        gt_mask = test_sample['mask'].to(device)
                        cur_batch = input_image.shape[0]

                        transformed_image = sam_model.preprocess(input_image)
                        input_shape = tuple(input_image.shape[-2:])

                        ################=================== Forward ===================################
                        image_embedding = sam_model.image_encoder(transformed_image)

                        prompt_box = torch.tensor(
                            [0, 0, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size], 
                            dtype=torch.float, 
                            device=device
                        )
                        prompt_box = prompt_box.tile((cur_batch, 1))

                        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                            points=None,
                            boxes=prompt_box,
                            masks=None,
                        )
                        
                        low_res_masks, iou_predictions = sam_model.mask_decoder(
                            image_embeddings=image_embedding,
                            image_pe=sam_model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                        )

                        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_shape, input_ori_shape).to(device)
                        # set negative values to 0
                        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
                        gt_binary_mask = torch.as_tensor(gt_mask, dtype=torch.float32)

                        loss = loss_fn(binary_mask, gt_binary_mask)

                        test_loss += loss.item()

                        pred_out_masks = torch.as_tensor(binary_mask[0], dtype=torch.uint8) * 255
                        pred_out_masks = pred_out_masks.cpu().numpy()
                        os.makedirs(os.path.join(exp_path, 'details', 'test'), exist_ok=True)
                        cv2.imwrite(os.path.join(exp_path, 'details', 'test', f't{iteration_count}_' + test_sample['name'][0]), pred_out_masks.squeeze())
                test_loss /= len(test_dataloader)
                
                with open(os.path.join(exp_path, 'details', 'test', 'test_errs.txt'), 'a') as log_file:  
                    log_file.write(str(test_loss) + '\n')

            if iteration_count > 500:
                return
            ######################################## DELETE THIS!!!
        # End epoch
            
        if e % save_epoch_interval == 0:
            save_checkpoint(iteration_count)
            last_ckpt_saved_epoch = e
        
        ################# test loop #################
        if with_test_dataset:
            test_loss = 0
            test_stats = np.array([0, 0, 0, 0])     # tp, tn, fp, fn
            for j, test_sample in enumerate(test_dataloader):
                with torch.no_grad():
                    cur_loss, cur_stats = forward_loop(test_sample, mode='test', cur_epoch=e)
                    test_loss += cur_loss
                    test_stats += cur_stats

            if not no_log_flag:
                wandb.log(
                    {
                        'test_loss': test_loss / len(test_dataloader),
                        'test_accuracy': (test_stats[0] + test_stats[1]) / np.sum(test_stats), 
                        'test_precision': test_stats[0] / (test_stats[0] + test_stats[2]), 
                        'test_recall': test_stats[0] / (test_stats[0] + test_stats[3])
                    }, 
                    step=iteration_count
                )
            tqdm.write(f'Test: [tp, tn, fp, fn] = {test_stats}')

        #wandb.log({'training_loss_epoch': mean(epoch_losses)}, step=e)

        tqdm.write(f'Train: Mean loss: {mean(epoch_losses)}')

    # End training
    if last_ckpt_saved_epoch != num_epoches:
        save_checkpoint(iteration_count)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('-e', '--epoch_num', type=int, default=10, help='max training epoches')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-n', '--exp_name', type=str, default='test', help='exp name. will be saved at exp/exp_name')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('-s', '--save_interval', type=int, default=2, help='checkpoint save interval (num of epoches)')
    parser.add_argument('--loss', type=int, default=0, help='TEST! 0 = MSE, 1 = BCE')
    parser.add_argument('--no_log', action='store_true', help='Do not log information to wandb')
    parser.add_argument('-d', '--data_dir', type=str)
    parser.add_argument('--test_dir', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()

    finetune(args)

    print('Finetuning complete!')
#mean_losses

#plt.plot(list(range(len(mean_losses))), mean_losses)
#plt.title('Mean epoch loss')
#plt.xlabel('Epoch Number')
#plt.ylabel('Loss')





'''

# Load up the model with default weights
sam_model_orig = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model_orig.to(device)



# Set up predictors for both tuned and original models
from segment_anything import sam_model_registry, SamPredictor
predictor_tuned = SamPredictor(sam_model)
predictor_original = SamPredictor(sam_model_orig)




# The model has not seen keys[21] (or keys[20]) since we only trained on keys[:20]
k = keys[21]
image = cv2.imread(f'scans/scans/{k}.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor_tuned.set_image(image)
predictor_original.set_image(image)

input_bbox = np.array(bbox_coords[k])

masks_tuned, _, _ = predictor_tuned.predict(
    point_coords=None,
    box=input_bbox,
    multimask_output=False,
)

masks_orig, _, _ = predictor_original.predict(
    point_coords=None,
    box=input_bbox,
    multimask_output=False,
)




#matplotlib inline
_, axs = plt.subplots(1, 2, figsize=(25, 25))


axs[0].imshow(image)
show_mask(masks_tuned, axs[0])
show_box(input_bbox, axs[0])
axs[0].set_title('Mask with Tuned Model', fontsize=26)
axs[0].axis('off')


axs[1].imshow(image)
show_mask(masks_orig, axs[1])
show_box(input_bbox, axs[1])
axs[1].set_title('Mask with Untuned Model', fontsize=26)
axs[1].axis('off')

plt.show()

'''

