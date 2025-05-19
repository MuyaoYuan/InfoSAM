import torch
import torch.nn as nn
import torch.nn.functional as F

# C = f(z1,z2)
# loss 1: I(z1,z2;C_t), compression
# loss 2: I(C_t;C_s), maximum mutual information
class DualMiLoss(nn.Module):
    def __init__(self, args):
        super(DualMiLoss, self).__init__()
        self.alpha = args.ib_alpha
        self.beta = args.ib_beta
        self.relation_type = args.relation_type
        self.max_epoch = args.epochs
        
    def compute_relation(self, feat, mask):
        dot_product = feat * mask 
        norm_feat = torch.norm(feat, dim=-1, keepdim=True)
        norm_mask = torch.norm(mask, dim=-1, keepdim=True)
        normalized_dot_product = dot_product / (norm_feat * norm_mask)
        relation = torch.sum(normalized_dot_product, dim=-1)
        relation = F.relu(relation)
        return relation
    
    def compute_relation_dot(self, feat, mask):
        batch_size = feat.shape[0]
        relation = feat @ mask.transpose(1, 2)  # (bsz, h*w)
        relation_norm = F.normalize(relation.view(batch_size,-1))
        return relation_norm

    def compute_Log(self, G_s):
        frobenius_norm_s = torch.norm(G_s, p='fro')  # ||G_s||_F
        frobenius_norm_s_squared = frobenius_norm_s ** 2  # ||G_s||_F^2
        log_frobenius_s = torch.log2(frobenius_norm_s_squared)  # log2(||G_s||_F^2)
        L_mi = log_frobenius_s
    
        return L_mi
    
    def compute_relation_loss(self, z1, z2, norm_f):
        # normlize
        batch_size = z1.shape[0]
        norm_z1 = F.normalize(z1.reshape(batch_size,-1))
        norm_z2 = F.normalize(z2.reshape(batch_size,-1))
        
        # compute gram matrix of z1, z2,
        G_z1 = torch.einsum('bx,dx->bd', norm_z1, norm_z1)
        G_z2 = torch.einsum('bx,dx->bd', norm_z2, norm_z2)
        G_f = torch.einsum('bx,dx->bd', norm_f, norm_f)
        G_tri =  G_z1 * G_z2 * G_f
        
        # Norm gram matrice
        G_f = G_f / torch.trace(G_f)
        G_tri = G_tri / torch.trace(G_tri)
        
        # compute log loss
        loss_f, loss_tri = self.compute_Log(G_f), self.compute_Log(G_tri)
        loss_r = - loss_f + loss_tri
        return loss_r
    
    def compute_distill_loss(self, norm_f_t, norm_f_s):
        # compute gram matrix of z1, z2,
        G_t = torch.einsum('bx,dx->bd', norm_f_t, norm_f_t)
        G_s = torch.einsum('bx,dx->bd', norm_f_s, norm_f_s)
        G_ts =  G_t * G_s
        
        # Norm gram matrice
        G_s = G_s / torch.trace(G_s)
        G_t = G_t / torch.trace(G_t)
        G_ts = G_ts / torch.trace(G_ts)
        
        # compute log loss
        loss_s, loss_t, loss_ts = self.compute_Log(G_s), self.compute_Log(G_t), self.compute_Log(G_ts)
        loss_d = loss_s + loss_t - loss_ts
        return loss_d
    
    def forward(self, student, teacher, relation_model=None, epoch=None, plot=False, relation_feature=None):
        """
        z_stu: size [batch_size, s_dim, h, w]
        z_tea: size [batch_size, t_dim, h, w]
        """
        feat_s, mask_s,_ = student
        feat_t, mask_t,_ = teacher
        
        if self.relation_type == 'attn':
            # compute the relation of feat and mask
            relation_t = relation_model(feat_t, mask_t) # [bsz, h*w]
            relation_s = relation_model(feat_s, mask_s) # [bsz, h*w]
        else:
            raise ValueError(f'No {self.relation_type}')
        
        # compute loss_r and loss_d
        loss_r = self.compute_relation_loss(z1=feat_t, z2=mask_t, norm_f=relation_t)
        loss_d = self.compute_distill_loss(norm_f_t=relation_t, norm_f_s=relation_s)
        
        loss = self.alpha * loss_r + self.beta * loss_d
        
        return loss