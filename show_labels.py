import os
import random
import numpy as np
import torch.nn.functional as F
from utils.general import *
from utils.plots import plot_images, plot_images_
from utils.datasets import LoadStreams, LoadImages, LoadImagesAndLabels, LoadImagesAndLabelsAndMasks
import cv2

from utils.torch_utils import init_torch_seeds
# names = ['security', 'uniform', 'person']
# names = ['uniform', 'no-uniform', 'hat', 'no-hat', 'trash-open', 'trash-close', 'play-phone', 'hand', 'glove']
# names = ['sleep', 'play_phone', 'phone']
# names = ['Workcard', 'Tie', 'Flower', 'Shortshirt', 'Longshirt', 'Suit', 'Shortjacket', 'Longjacket']
# names = ['cloth', 'hand', 'flower', 'workcard', 'tie', 'person']
# names = ['head', 'visible body', 'full body', 'motorcycle', 'small car', 'bicycle', 'unsure', 'baby carriage',
#          'midsize car', 'large car', 'electric car', 'tricycle']
names = ['full body']
seed = 2
random.seed(seed)
np.random.seed(seed)
init_torch_seeds(seed)
colors = [[random.randint(0, 255) for _ in range(3)]
          for _ in range(len(names))]

with open('data/hyp.scratch.yaml') as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

# dataset = LoadImagesAndLabels('data/play_phone1216/images/train', img_size=640, augment=True, cache_images=False,
#                               hyp=hyp)
dataset = LoadImagesAndLabelsAndMasks(
    '/d/baidubase/COCO/val_yolo/images/train',
    img_size=640,
    augment=True,
    cache_images=False,
    hyp=hyp,
)
dataset.mosaic = True

save = False
save_dir = '/d/projects/yolov5/data/play_phone0115/show_labels'

if save and not os.path.exists(save_dir):
    os.mkdir(save_dir)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    num_workers=0,
    sampler=None,
    pin_memory=True,
    collate_fn=LoadImagesAndLabelsAndMasks.collate_fn)
cv2.namedWindow('mosaic', cv2.WINDOW_NORMAL)
for i, (imgs, targets, paths, _, masks) in enumerate(dataloader):
    # for i, (imgs, targets, paths, _) in enumerate(dataset):
    #     print(targets)
    # print(targets)
    # if i in [4, 5]:
    if 1:
        result = plot_images_(images=imgs,
                              targets=targets,
                              paths=paths,
                              masks=masks)
        cv2.imshow('mosaic', result[:, :, ::-1])
        if cv2.waitKey(0) == ord('q'):  # q to quit
            break
        continue
    # imgs = imgs.numpy().astype(np.uint8).transpose((1, 2, 0))
    # imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    # targets = targets.numpy()
    # temp = targets[:, 2:6].copy()
    # targets[:, 2:6] = xywh2xyxy(temp * 960)
    # for idx, cls, *xyxy in targets:
    #     label = '%s' % (names[int(cls)])
    #     plot_one_box(xyxy, imgs, label=label, color=colors[int(cls)], line_thickness=2)
    # # print(targets[targets[:, 0] == 0])
    # if save and len(targets[targets[:, 1] == 0]) > 0:
    #     cv2.imwrite(os.path.join(save_dir, os.path.split(paths)[-1]), imgs)
    # cv2.imshow('p', imgs)
    # cv2.waitKey(0)
