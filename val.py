import torch
import numpy as np
import random
import os
import json
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from util.utils import FocalDiceloss_IoULoss, to_device
from util.val_util import show_anns
from datasets.dataset_csv import CSVDataset
from util.metrics import eval_seg, EvalSegSEF
import argparse
import matplotlib.pyplot as plt
from PIL import Image

def prompt_and_decoder(data_item, model, image_embeddings, image_size, multimask):
    if  data_item["point_coords"] is not None:
        points = (data_item["point_coords"], data_item["point_labels"])
    else:
        points = None


    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=points,
        boxes=data_item.get("boxes", None),
        masks=data_item.get("mask_inputs", None),
    )

    
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask,
    )
  
    if multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks,(image_size, image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions

def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )
    
    scale = image_size * 1.0 / max(ori_h, ori_w)
    newh, neww = ori_h * scale, ori_w * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    target_size = (newh, neww)
    masks = masks[..., : target_size[0], : target_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)

    return masks, scale


def parse_args():
    parser = argparse.ArgumentParser(description='Val')
    parser.add_argument('--data-path', type=str, help='Path to dataset folder')
    parser.add_argument('--sequence-path', type=str, help='Path to sequence folder')
    parser.add_argument('--save-path', type=str, default=None, help='Path for storing model snapshots')
    parser.add_argument('--backend', type=str, default='efficient_vit_h', help='Feature extractor')
    parser.add_argument('--snapshot', type=str, default='ckpt/sam_vit_h_4b8939.pth', help='Path to pretrained weights')
    parser.add_argument('--image_size', type=int, default=1024, help='Image size')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='leaf', help='dataset')
    parser.add_argument('--mask_num', type=int, default=5, help='number of sampling masks')
    parser.add_argument('--point_num', type=int, default=1, help='number of sampling point')
    parser.add_argument('--multimask', type=bool, default=True, help='ouput multimask')
    parser.add_argument('--no_multimask', dest='multimask', action='store_false')
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--val_workers", type=int, default=4)
    parser.add_argument('--down_sample', type=bool, default=True)
    parser.add_argument('--boxes_prompt', type=bool, default=True)
    parser.add_argument('--no_boxes_prompt', dest='boxes_prompt', action='store_false', help='Disable boxes_prompt')
    parser.add_argument("--metrics_type", type=int, default=1, help="metrics")
    parser.add_argument('--is_sam', type=bool, default=False)
    parser.add_argument('--sam_backend', type=str, default="vit_h")
    parser.add_argument('--sam_snapshot', type=str, default="ckpt/sam_vit_h_4b8939.pth")
    parser.add_argument('--split', type=str, default="val")
    parser.add_argument('--save_pred', type=bool, default=False)
    parser.add_argument('--save_gt', type=bool, default=False)
    parser.add_argument('--early_exit', type=int, default=10**18)
    parser.add_argument("--adapter_config", nargs='*', type=int, default=[-1,-1,-1,-1,-1,-1], help='The adapter config, \
                    len(config)==6 means only the third layer is adapted, len(config)==10 means all the layers except the first layer are adapted')
    parser.add_argument("--mlp_config", nargs='*', type=float, default=[0.25]*20, help='MLP ratio config')
    parser.add_argument("--rkd_type", type=str, default=None)

    parser.add_argument('--semantic_segmentation', action='store_true', default=False, help='Enable semantic segmentation (default: False)')

    return parser.parse_args()

def validation():

    args = parse_args()
    
    device = "cuda"
    os.makedirs(args.save_path, exist_ok=True)

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.metrics_type == 1:
        args.metrics = ['iou', 'dice']
    elif args.metrics_type == 2:
        args.metrics = ['Sm', 'Em', 'wFm'] # S-measure, E-measure, Weighted F-measure
        eval_seg_sef = EvalSegSEF()
    else:
        raise ValueError(f'no {args.metrics_type} metrics type')

    if args.is_sam:
        sam = sam_model_registry[args.sam_backend](checkpoint=args.sam_snapshot)
        sam.to(device=device)
        sam.eval()
    else:
        sam = None

    if not args.is_sam:
        efficient_sam = sam_model_registry[args.backend](checkpoint=args.snapshot, 
                                                    adapter_config=args.adapter_config,
                                                    mlp_config=args.mlp_config)
        efficient_sam.to(device=device)
        efficient_sam.eval()

    criterion = FocalDiceloss_IoULoss()

    binary_semantic_segmentation_datasets = ['leaf', 'road', 'isic', 'camo', 'sbu', 'kvasir']

    if args.dataset in binary_semantic_segmentation_datasets:
        test_ds = CSVDataset(args.data_path, mode=args.split,
                              csv_file=args.sequence_path,
                              image_size=args.image_size,
                              mask_num=args.mask_num,
                              point_num=args.point_num)
        test_ds_sa = test_ds
        assert args.mask_num == 1
    else:
        raise ValueError("Dataset is empty or not loaded correctly.")

    test_loader = DataLoader(dataset=test_ds_sa, batch_size=args.batch_size, shuffle=False, num_workers=args.val_workers, collate_fn=None)
    print('Test data:', len(test_loader))

    test_pbar = tqdm(test_loader)
    l = len(test_loader)

    test_loss = []
    test_res = [0] * len(args.metrics)
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    
    if args.save_pred:
        masks_for_image = []
        if args.save_gt:
            gt_for_image = []
    miou_stream = []
    for idx, data_item in enumerate(test_pbar):
        if idx > args.early_exit:
            print('early exit for showing')
            return
        
        if args.dataset in binary_semantic_segmentation_datasets:
            data_item = to_device(data_item, device)
            x, ori_labels, original_size = data_item["image"], data_item["ori_label"], data_item["original_size"]
            original_size = [original_size[0][0], original_size[1][0]]
        
        else:
            raise ValueError("Dataset is empty or not loaded correctly.")

        if idx == 0:
            last_image_path = data_item['image_path'][0]
        
        with torch.no_grad():
            if not args.is_sam:
                if "conv_lora" not in args.backend:
                    image_embeddings = efficient_sam.image_encoder(x)    
                else:
                    image_embeddings, all_moe_loss = efficient_sam.image_encoder(x)
                model = efficient_sam
            else:
                image_embeddings = sam.image_encoder(x)
                model = sam
                  

            if args.semantic_segmentation:
                data_item["point_coords"], data_item["point_labels"] = None, None
                data_item['boxes'] = None
            else:
                if args.boxes_prompt:
                    data_item["point_coords"], data_item["point_labels"] = None, None
                else:
                    data_item['boxes'] = None
            
            masks, low_res_masks, iou_predictions = prompt_and_decoder(data_item, model, image_embeddings, image_size=args.image_size, multimask=args.multimask)

        if args.dataset in binary_semantic_segmentation_datasets:
            masks, scale_post = postprocess_masks(low_res_masks, args.image_size, original_size)
            if args.save_pred:
                if data_item['image_path'][0] == last_image_path:
                    masks_for_image.append(masks)
                    last_image_path = data_item['image_path'][0]
                    if args.save_gt:
                        gt_for_image.append(ori_labels)
                else:
                    if args.dataset in binary_semantic_segmentation_datasets:
                        color = [1, 0.2, 0.6]
                    else:
                        color = None
                    dpi = 300
                    orig_image = Image.open(last_image_path)
                    plt.figure(figsize=(orig_image.width / dpi, orig_image.height / dpi), dpi=dpi)
                    plt.imshow(np.array(orig_image))
                    show_anns(torch.cat(masks_for_image, dim=0), color)
                    plt.axis('off')
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    mask_path = os.path.join(args.save_path, os.path.basename(f"{last_image_path}"))
                    plt.savefig(mask_path)
                    plt.close()

                    for idx, mask in enumerate(masks_for_image):
                        binary_mask = (mask.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8) * 255
                        individual_mask_path = os.path.join(
                            args.save_path, f"binary_{idx}_{os.path.basename(f'{last_image_path}')}"
                        )
                        Image.fromarray(binary_mask[0]).save(individual_mask_path)

                    if args.save_gt:
                        color = [1, 0, 0]
                        plt.figure(figsize=(orig_image.width / dpi, orig_image.height / dpi), dpi=dpi)
                        plt.imshow(np.array(orig_image))
                        show_anns(torch.cat(gt_for_image, dim=0), color)
                        plt.axis('off')
                        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                        label_path = os.path.join(args.save_path, 'gt_' + os.path.basename(f"{last_image_path}"))
                        plt.savefig(label_path)
                        plt.close()
                        gt_for_image = [ori_labels]
                    last_image_path = data_item['image_path'][0]
                    masks_for_image = [masks]
                     
            loss = criterion(masks, ori_labels, iou_predictions)
            test_loss.append(loss.item())

            if args.metrics_type == 1:
                temp = eval_seg(masks, ori_labels, threshold)
                test_res = [sum(a) for a in zip(test_res, temp)]
            elif args.metrics_type == 2:
                eval_seg_sef.step(masks, ori_labels)
        
        else:
            raise ValueError("Dataset is empty or not loaded correctly.")
        
        if args.metrics_type == 1:
            miou_stream.append(temp[0])
        
    if args.dataset in binary_semantic_segmentation_datasets:
        if args.metrics_type == 1:
            test_iter_metrics = [a/l for a in test_res]
            test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}
        elif args.metrics_type == 2:
            test_metrics = {args.metrics[i]: '{:.4f}'.format(list(eval_seg_sef.measures[i].get_results().values())[0]) for i in range(len(eval_seg_sef.measures))}
        average_loss = np.mean(test_loss)
        status = f"Test loss: {average_loss:.4f}, metrics: {test_metrics} \n"
        with open(os.path.join(args.save_path,"eval.txt"), "a") as f:
            f.write(status)
    else:
        raise ValueError("Dataset is empty or not loaded correctly.")
    
    with open(os.path.join(args.save_path,"miou_stream.json"), 'w') as miou_file:
        json.dump(miou_stream, miou_file)

    total_params = sum(p.numel() for p in model.parameters())

    with open(os.path.join(args.save_path,"parameters.txt"), "a") as f:
        f.write(f'parameters: {total_params} \n')

if __name__ == '__main__':
    validation()