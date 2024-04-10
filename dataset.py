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
        self.transform = ResizeLongestSide(self.size)
        self.images_dir_list = get_all_image_paths(os.path.join(data_path, 'imgs'))
        self.images_dir_list.sort()
        self.masks_dir_list = get_all_image_paths(os.path.join(data_path, 'masks'))
        self.masks_dir_list.sort()
        if len(self.images_dir_list) != len(self.masks_dir_list):
            raise Exception('Error: CableDataset: image and mask not match!')
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
        if with_augmentation:
            self.cable_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=1, hue=0.5)
            self.bg_transform = transforms.Compose([
                transforms.ColorJitter(brightness=(0.5, 1), contrast=0.2, saturation=0.2, hue=0),
                transforms.RandomResizedCrop(size=standard_size, ratio=(15 / 9, 17 / 9)), 
                transforms.GaussianBlur(kernel_size=5, sigma=(1.5, 5))
            ])


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

        if self.bg_dir_list is not None:
            bg_index = random.randint(0, len(self.bg_dir_list) - 1)
            bg = cv2.imread(self.bg_dir_list[bg_index])
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            if self.with_augmentation:
                image_torch = torch.from_numpy(image.transpose(2, 0, 1))
                bg_torch = torch.from_numpy(bg.transpose(2, 0, 1))
                image_torch = self.cable_transform(image_torch)
                bg_torch = self.bg_transform(bg_torch)
                mask_torch = torch.from_numpy(mask)
                image_torch = torch.where(mask_torch, image_torch, bg_torch)
                image = image_torch.numpy().transpose(1, 2, 0)
            else:
                bg = cv2.resize(image.shape[:2], interpolation=cv2.INTER_LINEAR)
                image = np.where(mask, image, bg)
        else:
            bg = None

        # Invert ?
        if self.invert:
            mask = ~mask

        transformed_image = self.transform.apply_image(image)
        transformed_image = torch.as_tensor(transformed_image, device='cpu')
        # (C, H, W)
        transformed_image = transformed_image.permute(2, 0, 1).contiguous()

        # add C channel -> (1, H, W)
        mask = torch.as_tensor(mask, device='cpu')[None, ...]

        return {
            'image': transformed_image, 
            'mask': mask, 
            'name': os.path.basename(self.images_dir_list[index])
        }
        

class MixedCableDataset(Dataset):
    def __init__(
            self, 
            datasets: List[CableDataset], 
            percentages: List[float], 
        ):
        if len(datasets) != len(percentages):
            raise Exception('Error: MixedDataset::__init__: lengths of datasets and percentages not match')
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
    
    bg_paths = ['dataset/SA-1B']
    train_dataset_1 = CableDataset(
        'dataset/refined_cable_dataset/train', 
        (720, 1280), 
        1024,
        background_data_paths=bg_paths,
        with_augmentation=True
    )

    train_dataset_2 = CableDataset(
        'dataset/p_cable_dataset/train', 
        (720, 1280), 
        1024
    )

    mixed_train_dataset = MixedCableDataset([train_dataset_1, train_dataset_2], [1, 0.5])

    train_dataloader = DataLoader(dataset=mixed_train_dataset, batch_size=1, shuffle=True)

    print(f'ori_shape: {train_dataset_1.standard_size}')
    for i, sample in enumerate(train_dataloader):
        input_image: np.ndarray = sample['image'].numpy()
        gt_mask: np.ndarray = sample['mask'].numpy()
        img_name = sample['name']

        print(f'current_shape: {input_image.shape}, name: {img_name}')
        
        cv2.imwrite(f'exp/dataset_test/output_img{i}.png', 
                    cv2.cvtColor(np.transpose(input_image.squeeze(), (1, 2, 0)), cv2.COLOR_RGB2BGR))

        if i > 64:
            break