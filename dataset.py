import os
from glob import glob
import numpy as np
import cv2
import torch
import random
from typing import Tuple, Literal, Dict, List, Any
from torch.utils.data import Dataset, DataLoader
from segment_anything.utils.transforms import ResizeLongestSide
import torchvision.transforms as transforms

def get_all_image_paths(directory):  
    valid_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.JPEG']  
    image_paths = []
      
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if os.path.splitext(file)[1] in valid_extensions:  
                image_paths.append(os.path.join(root, file))  
    
    return image_paths  


class CableDataset(Dataset):
    def __init__(
            self, 
            data_path: str, 
            standard_size: Tuple[int, int], 
            size: int, 
            background_data_paths: List[str] = None,
            with_augmentation: bool = False, 
            augmentation_details: List[str] = [], 
            invert: bool = False
        ):
        """
        Args:
            - data_path: cable dataset path, containing folders: "imgs" and "masks".
            - standard_size: (H, W), all images should be resized to this size first.
            - size: the longest side to be resized, using `ResizeLongestSide()`.
            - background_data_paths: a list of all data paths where the background may be sampled from.
                all .png / .jpg files in these directorys will be used randomly.
            - with_augmentation: whether to add data augmentation to cable images (and backgrounds).
                will be considered only of background_data_paths is specified.
            - invert: whether to invert g.t. mask
        """
        self.standard_size = standard_size
        self.size = size
        self.invert = invert

        # SAM provided transformation to resize the image and let the longest side = self.size
        self.transform = ResizeLongestSide(self.size)
        self.images_dir_list = get_all_image_paths(os.path.join(data_path, 'imgs'))
        self.images_dir_list.sort()
        self.masks_dir_list = get_all_image_paths(os.path.join(data_path, 'masks'))
        self.masks_dir_list.sort()
        if len(self.images_dir_list) != len(self.masks_dir_list):
            raise Exception('Error: CableDataset: image and mask do not match!')
        print('Found {} sample images / masks in directory "{}".'.format(len(self.images_dir_list), data_path))

        if background_data_paths is not None:
            self.bg_dir_list = []
            for bg_data_path in background_data_paths:
                self.bg_dir_list += get_all_image_paths(bg_data_path)
            self.bg_dir_list.sort()
            print(f'Found {len(self.bg_dir_list)} background images.')
        else:
            self.bg_dir_list = None

        self.with_augmentation = with_augmentation

        # transformations (data aug.)
        self.cable_transform = None         # data aug. on foreground cables (RGB image)
        self.foreground_transform = None    # data aug. on foreground (both RGB image and mask)
        self.bg_transform = None            # data aug. on background RGB image 
                                            # (or if there is no foreground, applied on the entire image)

        if with_augmentation:
            if self.bg_dir_list is not None:
                self.cable_transform = transforms.RandomApply(torch.nn.ModuleList([
                        transforms.ColorJitter(brightness=0.4, contrast=0.15, saturation=0.15, hue=0.5)
                    ]), p=0.5)
                self.bg_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=standard_size, ratio=(15 / 9, 17 / 9)), 
                    transforms.RandomApply(torch.nn.ModuleList([
                        transforms.ColorJitter(brightness=(0.5, 1), contrast=0.2, saturation=0.2, hue=0),
                        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 5))
                    ]), p=0.5)
                ])

                if 'crop_foreground' in augmentation_details:
                    self.foreground_transform = transforms.RandomApply(torch.nn.ModuleList([
                        transforms.RandomResizedCrop(
                            size=standard_size, 
                            scale=(0.35, 1), 
                            ratio=(standard_size[1] / standard_size[0], standard_size[1] / standard_size[0])
                        )
                    ]), p=0.5)

            else:
                self.bg_transform = transforms.RandomApply(torch.nn.ModuleList([
                    transforms.ColorJitter(brightness=0.1, contrast=0, saturation=0, hue=0.5)
                ]), p=0.5)


    def __len__(self) -> int:
        return len(self.images_dir_list)
    
    def __getitem__(self, index) -> Dict[Literal['image', 'mask', 'name'], Any]:
        """
        Returns:
            image, mask, original shapes of image and mask
        """
        image = cv2.imread(self.images_dir_list[index])
        mask = cv2.imread(self.masks_dir_list[index], cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.standard_size[::-1], interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.standard_size[::-1], interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0)

        mask_torch = torch.as_tensor(mask, dtype=torch.bool)

        if self.bg_dir_list is not None:
            bg_index = random.randint(0, len(self.bg_dir_list) - 1)
            bg = cv2.imread(self.bg_dir_list[bg_index])
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            if self.with_augmentation:
                image_torch = torch.from_numpy(image.transpose(2, 0, 1))
                bg_torch = torch.from_numpy(bg.transpose(2, 0, 1))
                image_torch = self.cable_transform(image_torch)
                bg_torch = self.bg_transform(bg_torch)
                if self.foreground_transform is not None:
                    img_mask_concat = torch.cat((image_torch, mask_torch[None, ...]), dim=0)
                    img_mask_concat = self.foreground_transform(img_mask_concat)
                    image_torch = img_mask_concat[:-1, ...]
                    mask_torch = img_mask_concat[-1, ...]
                mask_torch = mask_torch > 0
                image_torch = torch.where(mask_torch, image_torch, bg_torch)
                image = image_torch.numpy().transpose(1, 2, 0)
            else:
                bg = cv2.resize(image.shape[:2], interpolation=cv2.INTER_LINEAR)
                image = np.where(mask, image, bg)
        else:
            if self.with_augmentation:
                image_torch = torch.from_numpy(image.transpose(2, 0, 1))
                image_torch = self.bg_transform(image_torch)
                image = image_torch.numpy().transpose(1, 2, 0)
            bg = None

        # Invert ?
        if self.invert:
            mask_torch = ~mask_torch

        transformed_image = self.transform.apply_image(image)
        transformed_image = torch.as_tensor(transformed_image, device='cpu')
        # (C, H, W)
        transformed_image = transformed_image.permute(2, 0, 1).contiguous()

        # add C channel -> (1, H, W)
        mask_torch = mask_torch[None, ...]
        #mask = torch.as_tensor(mask, device='cpu')[None, ...]

        return {
            'image': transformed_image,         # torch.uint8
            'mask': mask_torch,                 # torch.bool
            'name': os.path.basename(self.images_dir_list[index])
        }
        

class MixedCableDataset(Dataset):
    def __init__(
            self, 
            datasets: List[CableDataset], 
            percentages: List[float], 
        ):
        if len(datasets) != len(percentages):
            raise Exception('Error: MixedDataset::__init__: lengths of datasets and percentages do not match')
        self.datasets = datasets
        self.counts = [int(len(datasets[i]) * percentages[i]) for i in range(len(datasets))]
        print(f'Mixed dataset: sample counts are {self.counts} respectively')
        self.resample()

    
    def __len__(self) -> int:
        return len(self.dataset_idxs)
    
    def __getitem__(self, index) -> Dict[Literal['image', 'mask', 'name'], Any]:
        dataset_idx = self.dataset_idxs[index]
        return self.datasets[dataset_idx[0]][dataset_idx[1]]

    def resample(self):
        """
        Resample the data indices in sub-datasets
        """
        self.dataset_idxs = []
        for i in range(len(self.datasets)):
            self.dataset_idxs += random.sample([(i, j) for j in range(len(self.datasets[i]))], self.counts[i])



if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib

    config_file_path = 'configs/train_balance8_40+50k_crop.yaml'
    def training_dataset_setup() -> DataLoader:
        import yaml

        with open(config_file_path) as f:
            config_yaml = yaml.load(f, Loader=yaml.FullLoader)

        dataset_config = config_yaml['dataset']

        train_data_subsets = []
        train_percentages = []
        for dataset_info in dataset_config['train']:
            new_train_dataset = CableDataset(
                data_path=dataset_info['data_dir'],
                standard_size=(720, 1280),
                size=1024,
                background_data_paths=dataset_info['bg_paths'],
                with_augmentation=dataset_info['w_aug'],
                augmentation_details=dataset_info['aug_details'] if (dataset_info['w_aug'] == True and 'aug_details' in dataset_info) else [], 
                invert=True
            )
            train_percentages.append(dataset_info['percentage'])
            train_data_subsets.append(new_train_dataset)
        
        train_dataset = MixedCableDataset(train_data_subsets, train_percentages)
        train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=1, 
            shuffle=True,
            num_workers=4, 
            pin_memory=True
        )
        return train_dataloader

    def custom_dataset_setup() -> DataLoader:
        #bg_paths = ['dataset/SA-1B']
        bg_paths = ['dataset/240417_rack_background']
        train_dataset_1 = CableDataset(
            'dataset/cable_synthetic/240412_data', 
            (720, 1280), 
            1024,
            background_data_paths=bg_paths,
            with_augmentation=True, 
            augmentation_details=['crop_foreground'], 
            invert=True
        )
        train_dataset_2 = CableDataset(
            'dataset/cable_synthetic/240417_data_iteration_2', 
            (720, 1280), 
            1024,
            background_data_paths=bg_paths,
            with_augmentation=True, 
            invert=True
        )
        train_dataset_3 = CableDataset(
            'dataset/cable_synthetic/240418_data_iteration_3', 
            (720, 1280), 
            1024,
            background_data_paths=bg_paths,
            with_augmentation=True, 
            augmentation_details=['crop_foreground'], 
            invert=True
        )

        print(f'ori_shape: {train_dataset_1.standard_size}')

        # train_dataset_2 = CableDataset(
        #     'dataset/p_cable_dataset/train', 
        #     (720, 1280), 
        #     1024
        # )

        #mixed_train_dataset = MixedCableDataset([train_dataset_1, train_dataset_2], [1, 0.5])
        mixed_train_dataset = MixedCableDataset([train_dataset_1, train_dataset_2, train_dataset_3], [0.5, 1, 1])

        train_dataloader = DataLoader(dataset=mixed_train_dataset, batch_size=1, shuffle=True)
        return train_dataloader

    train_dataloader = training_dataset_setup()

    for i, sample in enumerate(train_dataloader):
        input_image: np.ndarray = sample['image'].numpy()
        gt_mask: np.ndarray = sample['mask'].numpy()
        img_name = sample['name']

        print(f'current_shape: {input_image.shape}, name: {img_name}')
        
        cv2.imwrite(f'exp/dataset_sample_2/sample_img{i}.png', 
                    cv2.cvtColor(np.transpose(input_image.squeeze(), (1, 2, 0)), cv2.COLOR_RGB2BGR))

        if i >= 199:
            break