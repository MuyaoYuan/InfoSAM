import torch
import numpy as np
from torch.autograd import Function
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt as bwdist

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y


def iou(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()


def SegMetrics(pred, label, metrics):
    metric_list = []  
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric

# eval for batches 
def eval_seg(pred, true_mask_p, threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,1,h,w]
    pred: [b,1,h,w]
    '''
    b, c, h, w = pred.size()
    eiou, edice = 0,0
    for th in threshold:

        gt_vmask_p = (true_mask_p > th).float()
        vpred = (pred > th).float()
        vpred_cpu = vpred.cpu()
        disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

        disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')

        '''iou for numpy'''
        eiou += iou_b(disc_pred,disc_mask)

        '''dice for torch'''
        edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
        
    return eiou / len(threshold), edice / len(threshold)

def iou_b(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

_EPS = np.spacing(1)  # the different implementation of epsilon (extreme min value) between numpy and matlab
_TYPE = np.float64

def _prepare_data(pred, gt) -> tuple:
    """
    :param pred: prediction
    :param gt: mask
    :return: pred, gt
    """
    gt = gt.squeeze(0).squeeze(0).cpu().numpy() > 0.5
    pred = pred.squeeze(0).squeeze(0).cpu().numpy()
    # if pred.max() != pred.min():
    #     pred = (pred - pred.min()) / (pred.max() - pred.min())
    pred = np.clip(pred, 0, 1)
    return pred, gt

def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    """
    Return an adaptive threshold, which is equal to twice the mean of ``matrix``.
    :param matrix: a data array
    :param max_value: the upper limit of the threshold
    :return: min(2 * matrix.mean(), max_value)
    """
    return min(2 * matrix.mean(), max_value)

class Smeasure(object):
    def __init__(self, alpha: float = 0.5):
        """
        S-measure(Structure-measure) of SOD.
        ::
            @inproceedings{Smeasure,
                title={Structure-measure: A new way to eval foreground maps},
                author={Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Yun and Li, Tao and Borji, Ali},
                booktitle=ICCV,
                pages={4548--4557},
                year={2017}
            }
        :param alpha: the weight for balancing the object score and the region score
        """
        self.sms = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        sm = self.cal_sm(pred, gt)
        self.sms.append(sm)

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the S-measure.
        :return: s-measure
        """
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            sm = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
            sm = max(0, sm)
        return sm

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the object score.
        """
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        return object_score

    def s_object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x = np.mean(pred[gt == 1])
        sigma_x = np.std(pred[gt == 1], ddof=1)
        score = 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)
        return score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the region score.
        """
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info["weight"]
        # assert np.isclose(w1 + w2 + w3 + w4, 1), (w1 + w2 + w3 + w4, pred.mean(), gt.mean())

        pred1, pred2, pred3, pred4 = part_info["pred"]
        gt1, gt2, gt3, gt4 = part_info["gt"]
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def centroid(self, matrix: np.ndarray) -> tuple:
        """
        To ensure consistency with the matlab code, one is added to the centroid coordinate,
        so there is no need to use the redundant addition operation when dividing the region later,
        because the sequence generated by ``1:X`` in matlab will contain ``X``.
        :param matrix: a bool data array
        :return: the centroid coordinate
        """
        h, w = matrix.shape
        area_object = np.count_nonzero(matrix)
        if area_object == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            # More details can be found at: https://www.yuque.com/lart/blog/gpbigm
            y, x = np.argwhere(matrix).mean(axis=0).round()
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred: np.ndarray, gt: np.ndarray, x: int, y: int) -> dict:
        """
        Use (x,y) to divide the ``pred`` and the ``gt`` into four submatrices, respectively.
        """
        h, w = gt.shape
        area = h * w

        gt_LT = gt[0:y, 0:x]
        gt_RT = gt[0:y, x:w]
        gt_LB = gt[y:h, 0:x]
        gt_RB = gt[y:h, x:w]

        pred_LT = pred[0:y, 0:x]
        pred_RT = pred[0:y, x:w]
        pred_LB = pred[y:h, 0:x]
        pred_RB = pred[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3

        return dict(
            gt=(gt_LT, gt_RT, gt_LB, gt_RB),
            pred=(pred_LT, pred_RT, pred_LB, pred_RB),
            weight=(w1, w2, w3, w4),
        )

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the ssim score.
        """
        h, w = pred.shape
        N = h * w

        x = np.mean(pred)
        y = np.mean(gt)

        sigma_x = np.sum((pred - x) ** 2) / (N - 1)
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x**2 + y**2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + _EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def get_results(self) -> dict:
        """
        Return the results about S-measure.
        :return: dict(sm=sm)
        """
        sm = np.mean(np.array(self.sms, dtype=_TYPE))
        return dict(sm=sm)

class Emeasure(object):
    def __init__(self):
        """
        E-measure(Enhanced-alignment Measure) for SOD.
        More details about the implementation can be found in https://www.yuque.com/lart/blog/lwgt38
        ::
            @inproceedings{Emeasure,
                title="Enhanced-alignment Measure for Binary Foreground Map Evaluation",
                author="Deng-Ping {Fan} and Cheng {Gong} and Yang {Cao} and Bo {Ren} and Ming-Ming {Cheng} and Ali {Borji}",
                booktitle=IJCAI,
                pages="698--704",
                year={2018}
            }
        """
        self.adaptive_ems = []
        self.changeable_ems = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]

        changeable_ems = self.cal_changeable_em(pred, gt)
        self.changeable_ems.append(changeable_ems)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.adaptive_ems.append(adaptive_em)

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the adaptive E-measure.
        :return: adaptive_em
        """
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        adaptive_em = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)
        return adaptive_em

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Calculate the changeable E-measure, which can be used to obtain the mean E-measure,
        the maximum E-measure and the E-measure-threshold curve.
        :return: changeable_ems
        """
        changeable_ems = self.cal_em_with_cumsumhistogram(pred, gt)
        return changeable_ems

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        """
        Calculate the E-measure corresponding to the specific threshold.
        Variable naming rules within the function:
        ``[pred attribute(foreground fg, background bg)]_[gt attribute(foreground fg, background bg)]_[meaning]``
        If only ``pred`` or ``gt`` is considered, another corresponding attribute location is replaced with '``_``'.
        """
        binarized_pred = pred >= threshold
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)

        fg___numel = fg_fg_numel + fg_bg_numel
        bg___numel = self.gt_size - fg___numel

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel,
                fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg___numel,
                pred_bg_numel=bg___numel,
            )

            results_parts = []
            for i, (part_numel, combination) in enumerate(zip(parts_numel, combinations)):
                align_matrix_value = (
                    2 * (combination[0] * combination[1]) / (combination[0] ** 2 + combination[1] ** 2 + _EPS)
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts.append(enhanced_matrix_value * part_numel)
            enhanced_matrix_sum = sum(results_parts)

        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Calculate the E-measure corresponding to the threshold that varies from 0 to 255..
        Variable naming rules within the function:
        ``[pred attribute(foreground fg, background bg)]_[gt attribute(foreground fg, background bg)]_[meaning]``
        If only ``pred`` or ``gt`` is considered, another corresponding attribute location is replaced with '``_``'.
        """
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
        fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
        fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)

        fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        bg___numel_w_thrs = self.gt_size - fg___numel_w_thrs

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel_w_thrs
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel_w_thrs
        else:
            parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs,
                fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg___numel_w_thrs,
                pred_bg_numel=bg___numel_w_thrs,
            )

            results_parts = np.empty(shape=(4, 256), dtype=np.float64)
            for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
                align_matrix_value = (
                    2 * (combination[0] * combination[1]) / (combination[0] ** 2 + combination[1] ** 2 + _EPS)
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts[i] = enhanced_matrix_value * part_numel
            enhanced_matrix_sum = results_parts.sum(axis=0)

        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def generate_parts_numel_combinations(self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel

        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

        mean_pred_value = pred_fg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size

        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value

        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value),
        ]
        return parts_numel, combinations

    def get_results(self) -> dict:
        """
        Return the results about E-measure.
        :return: dict(em=dict(adp=adaptive_em, curve=changeable_em))
        """
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=_TYPE))
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=_TYPE), axis=0)
        # return dict(em=dict(adp=adaptive_em, curve=changeable_em))
        # unified output
        return dict(em=adaptive_em)

class WeightedFmeasure(object):
    def __init__(self, beta: float = 0.3):
        """
        Weighted F-measure for SOD.
        ::
            @inproceedings{wFmeasure,
                title={How to eval foreground maps?},
                author={Margolin, Ran and Zelnik-Manor, Lihi and Tal, Ayellet},
                booktitle=CVPR,
                pages={248--255},
                year={2014}
            }
        :param beta: the weight of the precision
        """
        self.beta = beta
        self.weighted_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        self.weighted_fms.append(wfm)

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the weighted F-measure.
        """
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)
        # Et = E;
        # Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        # MIN_E_EA = E;
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # %Pixel importance
        # B = ones(size(GT));
        # B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
        # Ew = MIN_E_EA.*B;
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        # TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
        # FPw = sum(sum(Ew(~GT)));
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])

        # R = 1- mean2(Ew(GT)); %Weighed Recall
        # P = TPw./(eps+TPw+FPw); %Weighted Precision
        # 注意这里使用mask索引矩阵的时候不可使用Ew[gt]，这实际上仅在索引Ew的0维度
        R = 1 - np.mean(Ew[gt == 1])
        P = TPw / (TPw + FPw + _EPS)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (R + self.beta * P + _EPS)

        return Q

    def matlab_style_gauss2D(self, shape: tuple = (7, 7), sigma: int = 5) -> np.ndarray:
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_results(self) -> dict:
        """
        Return the results about weighted F-measure.
        :return: dict(wfm=weighted_fm)
        """
        weighted_fm = np.mean(np.array(self.weighted_fms, dtype=_TYPE))
        return dict(wfm=weighted_fm)

class EvalSegSEF:
    def __init__(self):
        self.s_measure = Smeasure()
        self.e_measure = Emeasure()
        self.wf_measure = WeightedFmeasure()
        self.measures = [self.s_measure, self.e_measure, self.wf_measure]

    def step(self, pred, true_mask_p):

        '''
        threshold: a int or a tuple of int
        masks: [b,1,h,w]
        pred: [b,1,h,w]
        '''
        self.s_measure.step(pred, true_mask_p)
        self.e_measure.step(pred, true_mask_p)
        self.wf_measure.step(pred, true_mask_p)