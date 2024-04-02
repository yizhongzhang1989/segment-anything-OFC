import os
from glob import glob
import numpy as np
import cv2
import torch
from typing import Tuple, Literal, Dict, Any
from torch.utils.data import Dataset, DataLoader
from segment_anything.utils.transforms import ResizeLongestSide


class CableDataset(Dataset):
    def __init__(self, data_path: str, standard_size: Tuple[int, int], size: int):
        """
        Args:
            - standard_size: (H, W), all images should be resized to this size first.
            - size: the longest side to be resized, using `ResizeLongestSide()`.
        """
        self.standard_size = standard_size
        self.size = size
        self.transform = ResizeLongestSide(self.size)
        self.images_dir_list = glob(os.path.join(data_path, 'imgs', '*.png')) + glob(os.path.join(data_path, 'imgs', '*.jpg'))
        self.images_dir_list.sort()
        self.masks_dir_list = glob(os.path.join(data_path, 'masks', '*.png')) + glob(os.path.join(data_path, 'masks', '*.jpg'))
        self.masks_dir_list.sort()
        if len(self.images_dir_list) != len(self.masks_dir_list):
            raise Exception('Error: CableDataset: image and mask not match!')
        print('Found {} sample images / masks in directory "{}".'.format(len(self.images_dir_list), data_path))

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
        # Invert
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
        

if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib
    
    dataset_train_path = 'dataset/train/train'
    train_dataset = CableDataset(dataset_train_path, (720, 1280), 1024)

    print(f'ori_shape: {train_dataset.standard_size}')
    for i, sample in enumerate(train_dataset):
        input_image: np.ndarray = sample['image'].numpy()
        gt_mask: np.ndarray = sample['mask'].numpy()
        img_name = sample['name']

        print(f'current_shape: {input_image.shape}, name: {img_name}')
        
        cv2.imwrite('exp/test/output_img.png', np.transpose(input_image, (1, 2, 0)))

        if i > 32:
            break