import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import numpy as np
import random
import argparse
from segment_anything import sam_model_registry

from datasets.dataset_csv import CSVDataset
from util.utils import FocalDiceloss_IoULoss, StructureLoss, to_device, stack_dict_batched, check_unused_parameters,  get_total_grad_norm, cleanup_old_checkpoints

from torch.utils.tensorboard import SummaryWriter

from distiller.relation_model import RelationModule
from distiller.dualmi_distill import DualMiLoss


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


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data-path', type=str, help='Path to dataset folder')
    parser.add_argument('--sequence-path', type=str, help='Path to sequence folder', default='train.csv')
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--models-path', type=str, default=None, help='Path for storing model snapshots')
    parser.add_argument('--backend', type=str, default='efficient_vit_h', help='Feature extractor')
    parser.add_argument('--snapshot', type=str, default='ckpt/sam_vit_h_4b8939.pth', help='Path to pretrained weights')
    parser.add_argument('--image_size', type=int, default=1024, help='Image size')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
    parser.add_argument('--epochs', type=float, default=20, help='Number of training epochs to run')
    parser.add_argument('--start_lr', type=float, default=2e-4)
    parser.add_argument('--dataset', type=str, default='leaf', help='dataset')
    parser.add_argument('--resume', type=bool, default=False, help='Resume from snapshot')
    parser.add_argument('--mask_num', type=int, default=1, help='number of sampling masks')
    parser.add_argument('--point_num', type=int, default=1, help='number of sampling point')
    parser.add_argument('--multimask', type=bool, default=True, help='ouput multimask')
    parser.add_argument('--no_multimask', dest='multimask', action='store_false')
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument('--frozen', type=bool, default=True, help='frozen early layers or not')
    parser.add_argument("--train_workers", type=int, default=4)
    # learning rate scheduler
    parser.add_argument("--adapter_config", nargs='*', type=int, default=[-1,-1,-1,-1,-1,-1], help='The adapter config, \
                    len(config)==6 means only the third layer is adapted, len(config)==10 means all the layers except the first layer are adapted')
    parser.add_argument("--mlp_config", nargs='*', type=float, default=[0.25]*20, help='MLP ratio config')
    parser.add_argument("--unfrozen_norm", type=bool, default=False)
    
    # for retrain
    parser.add_argument("--starting_epoch", type=int, default=0)

    # rkd
    parser.add_argument("--rkd_type", type=str, default=None)
    parser.add_argument("--tuning_decoder", action="store_true", help="Enable tuning decoder.")
    parser.add_argument('--rkd_lambda', type=float, default=1)
    
    # ib
    parser.add_argument('--ib_alpha', type=float, default=1)
    parser.add_argument('--ib_beta', type=float, default=0.5)
    parser.add_argument('--relation_type', type=str, default=None)
    parser.add_argument('--backend_t', type=str, default='vit_b', help='backbone of teacher, e.g., vit_b, vit_t')
    parser.add_argument('--ckpt_t', type=str, default='ckpt/sam_vit_b_01ec64.pth', help='path of checkpoint of teacher')
    
    parser.add_argument('--semantic_segmentation', action='store_true', default=False, help='Enable semantic segmentation (default: False)')
    parser.add_argument('--prompt_type', type=float, default=0.5, help='0.0--only boxes, 1.0--only points')
    parser.add_argument('--gt_loss', type=str, default='FocalDiceloss_IoULoss')

    parser.add_argument('--moe_lambda', type=float, default=1, help='moe lambda for conv-lora')

    parser.add_argument("--BitFit", action="store_true", help="BitFit -- only finetune bias")

    parser.add_argument("--rm_ckpt", type=str, default=None)
    parser.add_argument("--rm_frozen", action="store_true", help="frozen relation model")

    return parser.parse_args()


def train():
    
    args = parse_args()

    os.makedirs(args.models_path, exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # tensorboard path
    tb_path = os.path.join(args.models_path, 'tensorboard_logs')
    writer = SummaryWriter(tb_path)

    if args.rkd_type is not None:
        sam = sam_model_registry[args.backend_t](checkpoint=args.ckpt_t)
        sam.to(device='cuda')
        sam.eval()
    else:
        sam = None

    if not args.resume:
        if args.backend in ['vit_b', 'vit_l', 'vit_h']:
            efficient_sam = sam_model_registry[args.backend](checkpoint=args.snapshot)
        else:
            efficient_sam = sam_model_registry[args.backend](checkpoint=args.snapshot,
                                                        adapter_config=args.adapter_config,
                                                        mlp_config=args.mlp_config)
        starting_epoch = 0
    else:
        checkpoint = torch.load(args.snapshot, map_location='cpu')
        if args.starting_epoch is not None:
            starting_epoch = int(args.starting_epoch)
        else:
            starting_epoch = checkpoint['epoch']

        efficient_sam = sam_model_registry[args.backend](checkpoint=args.snapshot,  
                                                    adapter_config=args.adapter_config,
                                                    mlp_config=args.mlp_config)
        print("Snapshot loaded from {}".format(args.snapshot))

    efficient_sam.to(device='cuda')
    efficient_sam.train()
    
    # init relation module
    if args.relation_type == 'attn':
        relation_model = RelationModule()
        if args.rm_ckpt is not None:
            with open(args.rm_ckpt, "rb") as f:
                state_dict = torch.load(f)
                relation_model.load_state_dict(state_dict['relation_model'], strict=True)
                print(f"Relation model load from {args.rm_ckpt}")
        relation_model.to(device='cuda')
        relation_model.train()
    else:
        relation_model = None
    
    
    # infoSAM
    if args.rkd_type == "dualmi":
        rkd_loss = DualMiLoss(args)
    else:
        raise ValueError(f'No {args.rkd_type}')
    
    for param in efficient_sam.parameters():
        param.requires_grad = False

    if args.frozen:
        # unfrozen adapter 
        adapter_count = 0
        try:
            for block in efficient_sam.image_encoder.blocks:
                for adapter in block.Adapters_list:
                    for param in adapter.parameters():
                        param.requires_grad = True
                    adapter_count += 1
        except:
            for lid, layer in enumerate(efficient_sam.image_encoder.layers):
                if lid == 0:
                    continue
                else:
                    for block in layer.blocks:
                        for adapter in block.Adapters_list:
                            for param in adapter.parameters():
                                param.requires_grad = True
                            adapter_count += 1
        print('adapter count: ', adapter_count)
        # unfrozen LoRA 
        if 'lora' in args.backend:
            lora_count = 0
            for n, p in efficient_sam.image_encoder.named_parameters():
                if 'lora' in n:
                    p.requires_grad = True
                    lora_count = lora_count + 1
            print('lora count: ', lora_count)
        
        # unfrozen norm layers when adapters exist
        if args.unfrozen_norm:
            norm_count = 0
            try:
                for block in efficient_sam.image_encoder.blocks:
                    try:
                        if len(block.Adapters_list) == 0:
                            continue
                    except:
                        pass
                    for module in block.modules():
                        if isinstance(module, nn.LayerNorm):
                            norm_count = norm_count + 1
                            for param in module.parameters():
                                param.requires_grad = True
            except:
                try:
                    for lid, layer in enumerate(efficient_sam.image_encoder.layers):
                        if lid == 0:
                            continue
                        else:
                            for block in layer.blocks:
                                try:
                                    if len(block.Adapters_list) == 0:
                                        continue
                                except:
                                    pass
                                for module in block.modules():
                                    if isinstance(module, nn.LayerNorm):
                                        norm_count = norm_count + 1
                                        for param in module.parameters():
                                            param.requires_grad = True
                except:
                    raise AttributeError
            print('unfrozen norm: ', norm_count)          

        if args.BitFit:
            bias_count = 0
            bias_list = []
            for i, block in enumerate(efficient_sam.image_encoder.blocks):
                for name, param in block.named_parameters():
                    if 'bias' in name:  
                        param.requires_grad = True
                        bias_count += 1
                        bias_list.append(name)
            print('unfrozen bias: ', bias_count)

    else:
        for param in efficient_sam.image_encoder.parameters():
            param.requires_grad = True

    if args.tuning_decoder:
        for n, p in efficient_sam.mask_decoder.named_parameters():
            p.requires_grad = True
        print('Apply full fine-tuning to the mask decoder')

    binary_semantic_segmentation_datasets = ['leaf', 'road', 'isic', 'camo', 'sbu', 'kvasir']

    if args.dataset in binary_semantic_segmentation_datasets:
        train_ds = CSVDataset(args.data_path, mode='train',
                              csv_file=args.sequence_path,
                              image_size=args.image_size,
                              mask_num=args.mask_num,
                              point_num=args.point_num)
        assert args.mask_num == 1
    else:
        raise ValueError("Dataset is empty or not loaded correctly.")

    sampler_train_sa = torch.utils.data.RandomSampler(train_ds)
    
    batch_sampler_train_sa = torch.utils.data.BatchSampler(
        sampler_train_sa, args.batch_size, drop_last=True)

    
    train_loader_sa = DataLoader(train_ds, batch_sampler=batch_sampler_train_sa,
                num_workers = args.train_workers,
                collate_fn=None,
                pin_memory = False)
    
    parameters = list(efficient_sam.parameters())
    if args.rkd_type == 'dualmi' and args.relation_type == 'attn':
        if args.rm_frozen:
            print('Relation model is forzen')
        else:
            parameters += list(relation_model.parameters())
    
    optimizer = optim.Adam(parameters, lr=args.start_lr)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*(len(train_ds) // args.batch_size + 1), eta_min=args.start_lr/10)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.gt_loss == 'FocalDiceloss_IoULoss':
        criterion = FocalDiceloss_IoULoss()
    elif args.gt_loss == 'StructureLoss':
        criterion = StructureLoss()
    else:
        raise ValueError(f'No {args.gt_loss}')
    
    loss_feat = torch.tensor(0)
    loss = torch.tensor(0)

    if args.epochs % 1 != 0:
        exit_idx = args.epochs*(len(train_ds) // args.batch_size + 1)
        args.epochs = 1
    else:
        exit_idx = 1e6
        args.epochs = int(args.epochs)
    
    step = 0
    plot=False
    for epoch in range(starting_epoch, args.epochs):
        epoch_losses = []
        epoch_feat_losses = []
        epoch_sa_losses = []
        train_iterator = train_loader_sa

        for idx, data_item in enumerate(train_iterator):
            
            step += 1

            if idx >= exit_idx:
                break
            
            optimizer.zero_grad()

            if args.dataset in binary_semantic_segmentation_datasets:
                data_item = stack_dict_batched(data_item)
                data_item = to_device(data_item, 'cuda')
                x = data_item["image"]
            else:
                raise ValueError("Dataset is empty or not loaded correctly.")
                
            x = x.cuda()
            
            if "conv_lora" not in args.backend:
                attn_features = efficient_sam.image_encoder(x)
            else:
                attn_features, all_moe_loss = efficient_sam.image_encoder(x)
         
            if args.semantic_segmentation:
                data_item['point_coords'] = None
                data_item['boxes'] = None
                flag = 'semantic_segmentation'
            else:
                if random.random() > args.prompt_type:
                    data_item['point_coords'] = None
                    flag = 'boxes'
                else:
                    data_item['boxes'] = None
                    flag = 'point'
            
            if args.dataset in binary_semantic_segmentation_datasets:
                batch, _, _, _ = attn_features.shape
                image_embeddings_repeat = []
                for i in range(batch):
                    image_embed = attn_features[i]
                    image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                    image_embeddings_repeat.append(image_embed)
                image_embeddings = torch.cat(image_embeddings_repeat, dim=0)
                
            else:
                raise ValueError("Dataset is empty or not loaded correctly.")
                                
                            
            masks, _, iou_predictions = prompt_and_decoder(data_item, efficient_sam, image_embeddings, args.image_size, args.multimask)

            labels = data_item["label"]
            if masks.shape[1] == 1:
                loss = criterion(masks, labels, iou_predictions)
            else: # the same label for multimask 
                loss = torch.tensor(0.).cuda()
                for multi_mask_id in range(masks.shape[1]):
                    loss =  loss + criterion(masks[:, multi_mask_id:multi_mask_id+1, ...], labels, iou_predictions)
            
            # save gt loss
            writer.add_scalar('gt Loss', loss.item(), step)
           
            if args.rkd_type is not None:
                # get student img_embedding, pred_mask, mask_token
                bs = image_embeddings.shape[0]
                image_embeddings = image_embeddings.permute(0, 2, 3, 1).reshape(bs, -1, 256)
                pred_masks = efficient_sam.mask_decoder.pred_masks
                mask_token = efficient_sam.mask_decoder.mask_tokens_output.unsqueeze(1)     
               
                # get teacher img_embedding, pred_mask, mask_token
                with torch.no_grad():
                    image_embeddings_t = sam.image_encoder(x)
                    mask_t, low_res_masks_t, iou_t  = prompt_and_decoder(data_item, sam, image_embeddings_t, args.image_size, args.multimask)
                image_embeddings_t = image_embeddings_t.permute(0, 2, 3, 1).reshape(bs, -1, 256)
                mask_token_t = sam.mask_decoder.mask_tokens_output.unsqueeze(1)
                pred_masks_t = sam.mask_decoder.pred_masks
               
                # compute the KD loss
                teacher = (image_embeddings_t, mask_token_t, pred_masks_t)
                student = (image_embeddings, mask_token, pred_masks)
                
                if args.rkd_type == 'dualmi':
                    loss_rkd = rkd_loss(student, teacher, relation_model, epoch, plot)
                    loss = loss + args.rkd_lambda *loss_rkd
                else:
                    raise ValueError(f'No {args.rkd_type}')
                
            if "conv_lora" in args.backend:
                loss = loss + args.moe_lambda * all_moe_loss
                writer.add_scalar('MoE Loss', all_moe_loss.item(), step)
            
            if args.rkd_type:
                writer.add_scalar('rkd Loss', loss_rkd.item(), step)
            
            writer.add_scalar('total Loss', loss.item(), step)
            
                    
            if torch.isnan(loss).any():
                raise ValueError("Loss contains NaN values. Taking corrective actions...")
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            # statistic grad
            grad_total_norm_2 = get_total_grad_norm(efficient_sam.parameters(), 2)

            if idx == 0:
                check_unused_parameters(efficient_sam)

            epoch_losses.append(loss.item())
            if args.rkd_type is not None:
                loss_feat = args.rkd_lambda * loss_rkd
            epoch_feat_losses.append(loss_feat.item())
            epoch_sa_losses.append(loss.item()-loss_feat.item())

            status_dict = {
                'loss': loss.item(),
                'avg': np.mean(epoch_losses),
                'avg_f': np.mean(epoch_feat_losses),
                'avg_s': np.mean(epoch_sa_losses),
            }

            status = '[{0}] loss = {1:0.5f} avg = {2:0.5f} avg_f = {3:0.5f} avg_s = {4:0.5f}, grad = {5:0.5f} LR = {6:0.7f} \n'.format(
                epoch + 1, status_dict['loss'], status_dict['avg'], status_dict['avg_f'], status_dict['avg_s'], grad_total_norm_2, scheduler.get_last_lr()[0])
            with open(os.path.join(args.models_path,"log_epoch.txt"), "a") as f:
                f.write(status)
        
        if (epoch + 1) % 1 == 0:
            torch.save(
                    {'model': efficient_sam.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict(),
                    'epoch': epoch,}, 
                    os.path.join(args.models_path, '_'.join([args.backend, str(epoch + 1), '.pth']))
            )
            cleanup_old_checkpoints(args.models_path, args.backend, epoch + 1, max_checkpoints=5)
            
            if args.relation_type == 'attn':
                rm_save_path = os.path.join(args.models_path, 'relation_models')
                os.makedirs(rm_save_path, exist_ok=True)
                torch.save(
                        {'relation_model': relation_model.state_dict(),
                        'relation_optimizer': optimizer.state_dict(),
                        'relation_lr_scheduler': scheduler.state_dict(),
                        'relation_epoch': epoch,}, 
                        os.path.join(rm_save_path,  f'{epoch + 1}.pth'))
    writer.close()

if __name__ == '__main__':
    train()