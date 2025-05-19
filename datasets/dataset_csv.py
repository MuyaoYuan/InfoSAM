import os
import csv
import cv2
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from util.utils import init_point_sampling, TransformSam, get_single_box_from_mask

class CSVDataset(data.Dataset):
    def __init__(self,
                 root_dir,
                 mode='train',
                 csv_file='train.csv',
                 image_size=1024,
                 mask_num=1, 
                 point_num=1,):
        """
        Args:
            root_dir (str): Root directory containing images and masks.
            mode (str): Mode of the dataset ('train' or 'val').
            csv_file (str): CSV file name containing image and mask paths.
        """
        super().__init__()
        self.root_dir = root_dir
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        print('self.mode', self.mode)
        self.data = []

        csv_path = os.path.join(root_dir, csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = os.path.join(root_dir, row['image'])
                label_path = os.path.join(root_dir, row['label'])
                self.data.append((image_path, label_path))


        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # natural image
            ])

        self.image_size = image_size
        self.mask_num = mask_num
        self.point_num = point_num

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data item.
        """

        image_input = {}

        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range")

        image_path, label_path = self.data[idx]

        image_ori = cv2.imread(image_path)
        mask_ori = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if mask_ori.max() == 255:
            mask_ori = mask_ori / 255
            mask_ori = mask_ori.astype(np.uint8)
        
        h_img, w_img, _ = image_ori.shape
        h_mask, w_mask = mask_ori.shape
        if (h_img != h_mask) or (w_img != w_mask):
            print(image_path, label_path)
            raise ValueError(f"Mismatch in dimensions: image ({h_img}, {w_img}) vs mask ({h_mask}, {w_mask})")
        
        transforms = TransformSam(self.image_size, h_img, w_img)
        image = transforms(image_ori)
        mask = transforms(mask_ori)

        # if self.mode == 'train':
        #     image, mask = self.aug_transforms(image, mask)

        image_tensor = self.to_tensor(image)
        mask_tensor = torch.from_numpy(mask).to(torch.int64).squeeze(0)

        boxes = get_single_box_from_mask(mask_tensor)
        point_coords, point_labels = init_point_sampling(mask_tensor, self.point_num)
        
        if self.mode == 'train':
            image_input["image"] = image_tensor.unsqueeze(0) #[1, 3, 1024, 1024]
            image_input["label"] = mask_tensor.unsqueeze(0).unsqueeze(0) # [mask_num, 1, 1024, 1024]
            image_input["boxes"] = boxes.unsqueeze(0) # [mask_num, 1, 4]
            image_input["point_coords"] = point_coords.unsqueeze(0) # [mask_num, 1, 2]
            image_input["point_labels"] = point_labels.unsqueeze(0) # [mask_num, 1]
            image_input["image_path"] = image_path
        elif self.mode in ('val', 'test'):
            image_input["image"] = image_tensor #[3, 1024, 1024]
            image_input["label"] = mask_tensor.unsqueeze(0) # [1, 1024, 1024]
            image_input["boxes"] = boxes # [1, 4]
            image_input["point_coords"] = point_coords # [1, 2]
            image_input["point_labels"] = point_labels # [1]
            image_input["original_size"] = (h_img, w_img)
            image_input["ori_label"] = torch.tensor(mask_ori).unsqueeze(0) # [1, h_img, w_img]
            image_input["image_path"] = image_path
            image_input["label_name"] = os.path.basename(label_path)

        return image_input