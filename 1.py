# import torch
# from utils.plots import plot_results
# import cv2
# import numpy as np
#
# a = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
#
# b = a[[0, 0, 0, 1, 2]]
# print(b)
#
# a = torch.tensor([0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 4, 5, 5, 6, 6, 7, 8])
# b = []
# index = -1
# ori = a[0]
# for i in range(len(a)):
#     if a[i] == ori:
#         index += 1
#         b.append(index)
#     else:
#         ori = a[i]
#         index = 0
#         b.append(index)
# print(b)
#
# a = np.random.randn(500, 500, 4)
# print(a.shape)
# b = cv2.resize(a, (250, 250))
# print(b.shape)
#
#
# def lettermask(mask,
#                new_shape=(640, 640),
#                color=(114, 114, 114),
#                auto=False,
#                scaleFill=False,
#                scaleup=True,
#                stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape, length = mask.shape[:2], mask.shape[
#         -1]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)
#
#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better test mAP)
#         r = min(r, 1.0)
#
#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
#         1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[
#             0]  # width, height ratios
#
#     dw /= 2  # divide padding into 2 sides
#     dh /= 2
#
#     if shape[::-1] != new_unpad:  # resize
#         mask = cv2.resize(mask, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     color = [0 for _ in range(length)]
#     mask = cv2.copyMakeBorder(mask,
#                               top,
#                               bottom,
#                               left,
#                               right,
#                               cv2.BORDER_CONSTANT,
#                               value=color)  # add border
#     return mask, ratio, (dw, dh)
#
#
# print(lettermask(a)[0].shape)
#
# a = torch.rand((32, 112))
# b = torch.rand((2, 32))
# print(a.shape)
# print(b.shape)
# # print(a @ b)
# print(torch.mm(b, a))
#
# # plot_results(save_dir='/d/projects/research/yolov5/runs/train/seg200/',
# #              mask_head=True)
#
# a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 8])
# b = a[a > 5][1]
# print(b)
# a[a > 5][1] = 1
# print(a)
#
# a = 0
# a |= 0
# print(a)
# print(1 | 0)

# import os
#
#
# os.system("python ./train_seg.py --weights weights/yolov5m.pt --cfg ./models/yolov5m_seg.yaml --name silu_m_test")
# os.system("python ./train_seg.py --weights weights/yolov5s.pt --cfg ./models/yolov5s_seg.yaml --name silu_s_test")

from utils.plots import plot_results
from utils.general import strip_optimizer
strip_optimizer('runs/train/coco_s/weights/last.pt')
# plot_results(save_dir='/d/projects/research/yolov5/runs/train/person_s', mask_head=True)


# import torch
#
# a = torch.randn(4, 5)[None, None, :]
# print(a.shape)
# print(a)





