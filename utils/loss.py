# Loss functions

import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np

from utils.general import bbox_iou, crop, xywh2xyxy
from utils.torch_utils import is_parallel


def smooth_BCE(
    eps=0.1
):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(
            reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t)**self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob)**self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get(
            'label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[
            -1]  # Detect() module
        self.balance = {
            3: [4.0, 1.0, 0.4]
        }.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(
            det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(
            1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p,
                                                          targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj,
                        gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2)**2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False,
                               CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj,
                     gi] = (1.0 -
                            self.gr) + self.gr * iou.detach().clamp(0).type(
                                tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn,
                                        device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[
                    i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(
            7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na,
                          device=targets.device).float().view(na, 1).repeat(
                              1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]),
                            2)  # append anchor indices
        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(
                    r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1),
                 gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def segment_loss(self, preds, targets,
                     masks):  # predictions, targets, model
        """
        proto_out:[batch-size, mask_dim, mask_hegiht, mask_width]
        masks:[batch-size * num_objs, image_height, image_width]
        每张图片objects数量不同，到时候处理时填充不足的
        """
        p = preds[0]
        proto_out = preds[1]
        mask_out = preds[2]
        # print(proto_out.shape)
        # batch_size, mask_dim, mask_h, mask_w
        mask_h, mask_w = proto_out.shape[2:]
        proto_out = proto_out.permute(0, 2, 3, 1)
        device = targets.device
        lcls, lbox, lobj, lseg = torch.zeros(1, device=device), torch.zeros(
            1, device=device), torch.zeros(1, device=device), torch.zeros(
                1, device=device)
        tcls, tbox, indices, anchors, tidxs, xywh = self.build_targets_(
            p, targets)  # targets

        # Losses
        # savei = 0
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj,
                        gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2)**2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False,
                               CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj,
                     gi] = (1.0 -
                            self.gr) + self.gr * iou.detach().clamp(0).type(
                                tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 37:], self.cn,
                                        device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 37:], t)  # BCE

                # mask proto
                # [mask_h, mask_w, mask_dim] @ [mask_dim, num_pos]
                # proto_temp = proto_out[b]
                # print(proto_temp.shape)
                # print(ps[:, 5:37].shape)
                # pred_mask = torch.clamp(
                #     proto_temp @ ps[:, 5:37].tanh().T,
                #     0,
                #     # mask_h, mask_w, num_pos
                #     1)
                # print(pred_mask.shape)
                # # num_pos, image_h, image_w
                mask_gt = masks[tidxs[i]]
                # print(len(mask_gt[mask_gt > 0]))
                downsampled_masks = F.interpolate(
                    mask_gt[None, :],
                    (mask_h, mask_w),
                    mode='nearest',
                )
                # mask_h, mask_w, num_pos
                # downsampled_masks = downsampled_masks.squeeze().permute(
                #     1, 2, 0).contiguous()
                # lseg += F.binary_cross_entropy(pred_mask,
                #                                downsampled_masks,
                #                                reduce='sum')
                # print(b.unique())
                for bi in b.unique():
                    # print(b, bi)
                    index = b == bi
                    bm, am, gjm, gim = b[index], a[index], gj[index], gi[index]
                    mask_gti = downsampled_masks.squeeze()[index]
                    mask_gti = mask_gti.permute(1, 2, 0).contiguous()
                    mxywh = xywh[i][index]
                    mw, mh = mxywh[:, 2:].T
                    mw, mh = mw / pi.shape[3], pi.shape[2]
                    # print(mxywh.shape)
                    mxywh = mxywh / torch.tensor(
                        pi.shape,
                        device=mxywh.device)[[3, 2, 3, 2]] * torch.tensor(
                            [mask_w, mask_h, mask_w, mask_h],
                            device=mxywh.device)  # psi = ps[b == bi]
                    mxyxy = xywh2xyxy(mxywh)
                    psi = pi[bm, am, gjm, gim]
                    # psi.tanh().cpu().detach().numpy())
                    pred_maski = proto_out[bi] @ psi[:, 5:37].tanh().T
                    # pred_maski = proto_out[bi] @ psi.tanh().T

                    # np.savetxt(
                    #     f'mask_c/{savei}.txt',
                    #     pred_maski[:, :, 0].sigmoid().cpu().detach().numpy())

                    # pred_maski = pred_maski.sigmoid()
                    # savei += 1
                    # print(pred_maski.shape)
                    # print(mask_gti.shape)
                    # print(len(mask_gti[mask_gti > 0]))
                    # cv2.imshow(
                    #     'p',
                    #     mask_gti[:, :, 0].cpu().numpy().astype(np.uint8) * 255)
                    # if cv2.waitKey(0) == ord('q'):
                    #     exit()
                    # lseg += nn.MSELoss(reduction='mean')(pred_maski, mask_gti)
                    lseg_ = F.binary_cross_entropy_with_logits(
                        pred_maski, mask_gti, reduction='none')

                    lseg_ = crop(lseg_, mxyxy)
                    # print(lseg_.shape)
                    lseg_ = lseg_.mean(dim=(0, 1)) / mw / mh
                    lseg += torch.mean(lseg_)

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[
                    i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lseg *= self.hyp['box'] * 10
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lseg
        return loss * bs, torch.cat((lbox, lobj, lcls, lseg, loss)).detach()

    def build_targets_(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch, tidxs, xywh = [], [], [], [], [], []
        gain = torch.ones(
            8, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na,
                          device=targets.device).float().view(na, 1).repeat(
                              1, nt)  # same as .repeat_interleave(nt)

        ti = torch.arange(nt,
                          device=targets.device).float().view(1, nt).repeat(
                              na, 1)  # same as .repeat_interleave(nt)
        # batch_idx = targets[:, 0]
        # b = []
        # index = -1
        # ori = batch_idx[0]
        # for i in range(len(batch_idx)):
        #     if a[i] == ori:
        #         index += 1
        #         b.append(index)
        #     else:
        #         ori = a[i]
        #         index = 0
        #         b.append(index)
        targets = torch.cat(
            (targets.repeat(na, 1, 1), ai[:, :, None], ti[:, :, None]),
            2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(
                    r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            tidx = t[:, 7].long()
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1),
                 gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            tidxs.append(tidx)
            xywh.append(torch.cat((gxy, gwh), 1))

        return tcls, tbox, indices, anch, tidxs, xywh
