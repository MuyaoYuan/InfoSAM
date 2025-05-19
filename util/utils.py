import torch
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
import os
import glob
from torch.nn.parallel import DistributedDataParallel
from torchvision.transforms.functional import resize, to_pil_image 

# function for grid generation
def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def batch_iterator(batch_size: int, *args):
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

def cleanup_old_checkpoints(models_path, backend, epoch, max_checkpoints=5):
    checkpoint_pattern = os.path.join(models_path, f"{backend}_*.pth")
    checkpoint_files = sorted(glob.glob(checkpoint_pattern), key=os.path.getmtime)

    if len(checkpoint_files) > max_checkpoints:
        for file_to_remove in checkpoint_files[:-max_checkpoints]:
            os.remove(file_to_remove)

def get_total_grad_norm(parameters, norm_type=2):
    
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)

    return total_norm


def get_single_box_from_mask(mask, std=0.1, max_pixel=5):
    """
    Args:
        mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_box: A single bounding box with noise perturbation, returned as a torch.Tensor.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    # Find the coordinates of the non-zero elements in the mask
    non_zero_coords = np.argwhere(mask > 0)

    if non_zero_coords.size == 0:
        # Return an empty box if the mask is empty
        return torch.tensor([0, 0, 0, 0], dtype=torch.float)

    # Calculate the bounding box that wraps the entire mask
    y_min, x_min = non_zero_coords.min(axis=0)
    y_max, x_max = non_zero_coords.max(axis=0)

    # Perturb the bounding box with noise
    width, height = x_max - x_min, y_max - y_min
    noise_std = min(width, height) * std
    max_noise = min(max_pixel, int(noise_std * 5))

    # Add random noise to each coordinate
    noise_x = np.random.randint(-max_noise, max_noise)
    noise_y = np.random.randint(-max_noise, max_noise)

    x_min = max(0, x_min + noise_x)  # Ensure the coordinates are non-negative
    y_min = max(0, y_min + noise_y)
    x_max = max(0, x_max + noise_x)
    y_max = max(0, y_max + noise_y)

    # Return the perturbed bounding box as a torch.Tensor
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float).unsqueeze(0)


def select_random_points(pr, gt, point_num = 9):
    """
    Selects random points from the predicted and ground truth masks and assigns labels to them.
    Args:
        pred (torch.Tensor): Predicted mask tensor.
        gt (torch.Tensor): Ground truth mask tensor.
        point_num (int): Number of random points to select. Default is 9.
    Returns:
        batch_points (np.array): Array of selected points coordinates (x, y) for each batch.
        batch_labels (np.array): Array of corresponding labels (0 for background, 1 for foreground) for each batch.
    """
    pred, gt = pr.data.cpu().numpy(), gt.data.cpu().numpy()
    error = np.zeros_like(pred)
    error[pred != gt] = 1

    # error = np.logical_xor(pred, gt)
    batch_points = []
    batch_labels = []
    for j in range(error.shape[0]):
        one_pred = pred[j].squeeze(0)
        one_gt = gt[j].squeeze(0)
        one_erroer = error[j].squeeze(0)

        indices = np.argwhere(one_erroer == 1)
        if indices.shape[0] > 0:
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        else:
            indices = np.random.randint(0, 256, size=(point_num, 2))
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        selected_indices = selected_indices.reshape(-1, 2)

        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]
            if one_pred[x,y] == 0 and one_gt[x,y] == 1:
                label = 1
            elif one_pred[x,y] == 1 and one_gt[x,y] == 0:
                label = 0
            points.append((y, x))   #Negate the coordinates
            labels.append(label)

        batch_points.append(points)
        batch_labels.append(labels)
    return np.array(batch_points), np.array(batch_labels)


def init_point_sampling(mask, get_point=1):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
     # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:,::-1]
    bg_coords = np.argwhere(mask == 0)[:,::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices], dtype=torch.int)
        return coords, labels

class TransformSam():
    def __init__(self, img_size=1024, ori_h=720, ori_w=960) -> None:

        self.img_size = img_size
        self.ori_h = ori_h
        self.ori_w = ori_w
        
    def __call__(self, image, ):
        """
        the transform is the same as SAM, please refer to:
        segment_anything/modeling/sam.py -- preprocess()
        segment_anything/utils/transforms.py -- ResizeLongestSide.get_preprocess_shape()
        """
        scale = self.img_size * 1.0 / max(self.ori_h, self.ori_w)
        newh, neww = self.ori_h * scale, self.ori_w * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        target_size = (newh, neww)
        # resize to (768,1024)
        image = np.array(resize(to_pil_image(image), target_size))
        # padding to (1024,1024)
        padh = self.img_size - newh
        padw = self.img_size - neww
        if len(image.shape) == 3:
            image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode='constant')    
        else:
            image = np.expand_dims(image, axis=2)
            image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode='constant')
            image = np.squeeze(image, axis=2)
        return image


#Loss funcation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss


class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):
    
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        # replace focal loss with BCE loss
        # self.focal_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        # loss = loss1
        return loss

class StructureLoss(nn.Module):
    """
    The loss represent the weighted IoU loss and binary cross entropy (BCE) loss for the global restriction and local (pixel-level) restriction.

    References:
        [1] https://arxiv.org/abs/2006.11392
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor, pred_iou):
        if input.dim() == 3:
            input = input.unsqueeze(1)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = F.binary_cross_entropy_with_logits(input, target, reduce="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        input = torch.sigmoid(input)
        inter = ((input * target) * weit).sum(dim=(2, 3))
        union = ((input + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

def stack_dict_batched(data_item):
    out_dict = {}
    for k,v in data_item.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict

def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input

def unwrap(wrapped_module):
    if isinstance(wrapped_module, DistributedDataParallel):
        module = wrapped_module.module
    else:
        module = wrapped_module
    return module

def check_unused_parameters(model, exclude_keywords=["iou_prediction_head", "lora_moe_experts"]):
    unused_params = [name for name, param in unwrap(model).named_parameters()
                     if param.grad is None and param.requires_grad and not any(keyword in name for keyword in exclude_keywords)]
    
    if unused_params:
        raise RuntimeError(f"Unused parameters: {unused_params}")
    else:
        print("All the parameters are used.")