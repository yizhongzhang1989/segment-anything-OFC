import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from typing import Tuple, Literal, Dict, Any
from dataset import CableDataset, MixedCableDataset
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

BUSY_TEST = os.getenv('BUSY_TEST_ON', 'False').lower() in ('true', '1')
print('Busy test mode: ' + ('ON' if BUSY_TEST is True else 'OFF'))


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        if m.kernel_size == 1:
            nn.init.constant_(m.weight, 1)
        else:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):  
        nn.init.constant_(m.weight, 1)  
        nn.init.constant_(m.bias, 0)
        

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
    test_iter_interval = args.test_interval
    exp_path = os.path.join('exp', args.exp_name)   #'exp/finetune_first_test'
    batch_size = args.batch_size
    virtual_batch_size = args.virtual_batch_size
    true_batch_size = batch_size if virtual_batch_size is None else batch_size * virtual_batch_size
    device = args.device
    predefined_cable_alpha = args.alpha
    no_log_flag = args.no_log
    no_inv_dataset = args.no_inv
    is_full_finetune = args.full_tune
    is_param_init = args.init_param

    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'details'), exist_ok=True)

    # Set SAM model
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    if is_param_init:
        sam_model.mask_decoder.apply(init_weights)
    sam_model.to(device)
    sam_model.train()


    import yaml
    import shutil
    shutil.copy(args.config, os.path.join(exp_path, 'details', 'training_config.yaml'))
    with open(args.config) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)

        dataset_config = config_yaml['dataset']

        print(f'image_encoder.img_size: {sam_model.image_encoder.img_size}')

        train_data_subsets = []
        train_percentages = []
        for dataset_info in dataset_config['train']:
            new_train_dataset = CableDataset(
                data_path=dataset_info['data_dir'],
                standard_size=input_ori_shape,
                size=sam_model.image_encoder.img_size,
                background_data_paths=dataset_info['bg_paths'],
                with_augmentation=dataset_info['w_aug'],
                invert=not no_inv_dataset
            )
            train_percentages.append(dataset_info['percentage'])
            train_data_subsets.append(new_train_dataset)
        
        train_dataset = MixedCableDataset(train_data_subsets, train_percentages)
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4, 
            pin_memory=True
        )

        if 'test' in dataset_config and dataset_config['test'] is not None:
            with_test_dataset = True
            test_dataset = CableDataset(
                data_path=dataset_config['test']['data_dir'],
                standard_size=input_ori_shape,
                size=sam_model.image_encoder.img_size,
                background_data_paths=dataset_config['test']['bg_paths'],
                with_augmentation=dataset_config['test']['w_aug'],
                invert=not no_inv_dataset
            )

            test_dataloader = DataLoader(
                dataset=test_dataset, 
                batch_size=max(1, batch_size // 4), 
                shuffle=False,
                num_workers=4, 
                pin_memory=True
            )
        else:
            with_test_dataset = False


    ########################
    # Loss function choosing
    
    # no_inv means: 1=cable, 0=background  =>  gt=1 : alpha=loss=0.85
    if predefined_cable_alpha is None:
        loss_alpha = 0.85 if no_inv_dataset else 0.15
    else:
        loss_alpha = predefined_cable_alpha if no_inv_dataset else (1 - predefined_cable_alpha)
    focal_loss_fn = FocalLoss(alpha=loss_alpha, gamma=2, reduction='mean')
    dice_loss_fn = DiceLoss()
    loss_fn = lambda x, y: 20.0 * focal_loss_fn(x, y) + 1.0 * dice_loss_fn(x, y)
    #loss_fn = None
    #if args.loss == 0:
    #    loss_fn = torch.nn.MSELoss()
    #elif args.loss == 1:
    #    loss_fn = torch.nn.BCELoss()
    ########################


    

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
        log_str = \
f'''
Training details:
    learning rate (initial, after warm-up): {init_lr} 
    max epoch: {num_epoches} 
    batch_size: {batch_size} 
    virtual_batch: {'None' if virtual_batch_size is None else f'size = {virtual_batch_size}'}
    checkpoint save interval (num of epoches): {save_epoch_interval} 
    test interval (num of iterations): {test_iter_interval} 
    device: {device} 
    finetune: {'image encoder and mask decoder' if is_full_finetune else 'mask decoder only'}
    randomize parameters of mask decoder: {is_param_init}
    Invert dataset so that g.t. cables = 0 : {not no_inv_dataset}
    scheduler: {lr_scheduler}
        milestones: {lr_scheduler.milestones}
        gamma: {lr_scheduler.gamma}
Loss:
    loss funtion: {loss_fn}
    focal loss alpha (when g.t = 1): {loss_alpha}
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
            mode: Literal['train', 'test', 'busy_test'] = 'train', 
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
        # full finetune
        if is_full_finetune:
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
        # No grad here as we don't want to optimise the encoders
        else:
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

        upscaled_mask_logits = sam_model.postprocess_masks(low_res_masks, input_shape, input_ori_shape).to(device)
        gt_binary_mask = torch.as_tensor(gt_mask, dtype=torch.float32)
        loss = loss_fn(upscaled_mask_logits, gt_binary_mask)

        #####
        if DEBUG_MODE:
            binary_mask = upscaled_mask_logits > 0
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
            plt.close()
        ################=================== Forward ===================################

        tp, tn, fp, fn = 0, 0, 0, 0

        if mode == 'train':
            if virtual_batch_size is not None:
                loss /= virtual_batch_size
                loss.backward()
                if iteration_count % virtual_batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    if not no_log_flag:
                        wandb.log({'training_loss': loss.item() * virtual_batch_size}, step=iteration_count // virtual_batch_size)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                if not no_log_flag:
                    wandb.log({'training_loss': loss.item()}, step=iteration_count)

            iter_bar.update(cur_batch)
            iter_bar.set_description('[epoch:{}, iter:{}, loss:{:.4f}, lr:{:.6f}]'.format(cur_epoch, iteration_count, loss.item(), optimizer.param_groups[0]['lr']))

            
        elif mode == 'test':
            pred_bool_masks = upscaled_mask_logits > 0
            gt_bool_masks = torch.as_tensor(gt_mask, dtype=torch.bool)
            tp = torch.sum(pred_bool_masks & gt_bool_masks).item()
            tn = torch.sum(~pred_bool_masks & ~gt_bool_masks).item()
            fp = torch.sum(pred_bool_masks & ~gt_bool_masks).item()
            fn = torch.sum(~pred_bool_masks & gt_bool_masks).item()
        elif mode == 'busy_test':
            pred_bool_masks = upscaled_mask_logits > 0
            gt_bool_masks = torch.as_tensor(gt_mask, dtype=torch.bool)
            tp = torch.sum(pred_bool_masks & gt_bool_masks).item()
            tn = torch.sum(~pred_bool_masks & ~gt_bool_masks).item()
            fp = torch.sum(pred_bool_masks & ~gt_bool_masks).item()
            fn = torch.sum(~pred_bool_masks & gt_bool_masks).item()
            
            pred_prob_map = torch.sigmoid(upscaled_mask_logits[0])
            pred_prob_map_out: np.ndarray = pred_prob_map.cpu().numpy() * 255
            pred_prob_map_out = pred_prob_map_out.astype(np.uint8)
            pred_bin_mask_out: np.ndarray = pred_bool_masks[0].cpu().numpy().astype(np.uint8) * 255
            cv2.imwrite(os.path.join(exp_path, 'details', 'busy_test', 'probs', f't{iteration_count}_' + test_sample['name'][0]), pred_prob_map_out.squeeze())
            cv2.imwrite(os.path.join(exp_path, 'details', 'busy_test', 'bin_masks', f't{iteration_count}_' + test_sample['name'][0]), pred_bin_mask_out.squeeze())
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
        
        # Resample sub-datasets
        train_dataset.resample()

        # Iterate over the dataset
        for i, train_sample in enumerate(train_dataloader):
            cur_loss, _ = forward_loop(train_sample, mode='train', cur_epoch=e)
            epoch_losses.append(cur_loss)
            all_losses.append(cur_loss)

            ######################################## TEMPERARY TEST !
            if BUSY_TEST:
                os.makedirs(os.path.join(exp_path, 'details', 'busy_test', 'probs'), exist_ok=True)
                os.makedirs(os.path.join(exp_path, 'details', 'busy_test', 'bin_masks'), exist_ok=True)

                if iteration_count > 1000:
                    return
            ######################################## TEMPERARY TEST !

            ################# test loop #################
            if with_test_dataset:
                if (iteration_count % test_iter_interval == 0) or (BUSY_TEST and iteration_count % 20 == 0):
                    test_loss = 0
                    test_stats = np.array([0, 0, 0, 0])     # tp, tn, fp, fn
                    test_mode = 'busy_test' if BUSY_TEST else 'test' 
                    for j, test_sample in enumerate(test_dataloader):
                        with torch.no_grad():
                            cur_loss, cur_stats = forward_loop(test_sample, mode=test_mode, cur_epoch=e)
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

                    if BUSY_TEST:
                        with open(os.path.join(exp_path, 'details', 'busy_test', 'test_errs.txt'), 'a') as log_file:  
                            log_file.write(str(test_loss) + '\n')
            ################## End test loop
        # End epoch
            
        if e % save_epoch_interval == 0:
            save_checkpoint(iteration_count)
            last_ckpt_saved_epoch = e
        

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
    parser.add_argument('--virtual_batch_size', type=int, default=None, 
                        help='virtual batch size. True batch size = batch_size * virtual_batch_size. Determines the gradient clear interval')
    parser.add_argument('-n', '--exp_name', type=str, default='test', help='exp name. will be saved at exp/exp_name')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('-s', '--save_interval', type=int, default=2, help='checkpoint save interval (num of epoches)')
    parser.add_argument('--test_interval', type=int, default=200, help='test interval (num of iterations, "test_dir" must be specified)')
    parser.add_argument('--no_log', action='store_true', help='Do not log information to wandb')
    parser.add_argument('--full_tune', action='store_true', help='Enable to finetune both encoder and decoder')
    parser.add_argument('--init_param', action='store_true', help='Enable to randomly initialize parameters of mask decoder')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha value (0, 1) for focal loss where there are cables')
    parser.add_argument('--no_inv', action='store_true', help='Do not invert dataset masks')
    parser.add_argument('-c', '--config', type=str, default=None, help='Config path')
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

