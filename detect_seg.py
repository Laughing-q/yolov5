import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression_, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, process_mask, scale_masks, process_mask_upsample
from utils.plots import plot_one_box, plot_one_mask
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np

np.set_printoptions(threshold=10000000)


def detect(save_img=False):
    pause = True
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith(
        '.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith(
        '.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(
        increment_path(Path(opt.project) / opt.name,
                       exist_ok=opt.exist_ok))  # increment run
    if save_img or save_txt:
        (save_dir / 'labels' if save_txt else save_dir).mkdir(
            parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(
            torch.load('weights/resnet101.pt',
                       map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        view_img = check_imshow()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred, _, proto_out = model(img, augment=opt.augment)

        # Apply NMS
        pred = non_max_suppression_(pred,
                                    opt.conf_thres,
                                    opt.iou_thres,
                                    classes=opt.classes,
                                    agnostic=opt.agnostic_nms,
                                    mask_out=[])
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1,
                                          0]]  # normalization gain whwh
            if len(det):
                # print('det:', det.shape)
                out_masks = det[:, 6:]  # [img_h, img_w, num]
                masks = process_mask_upsample(proto_out[i], out_masks, det[:, :4],
                                     img.shape[2:])
                masks = scale_masks(img.shape[2:], masks, im0.shape)
                # print(masks.shape)
                # np.save(p.name + '.npy', masks.cpu().numpy())
                # for mi in masks.cpu().numpy():
                #     # print(mi.max())
                #     # print(mi.min())
                #     # print(mi[mi > 0])
                #     # exit()
                #     cv2.imshow('p', mi * 255)
                #     if cv2.waitKey(0) == ord('q'):
                #         exit()
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for i, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                                gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (
                            cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy,
                                     im0,
                                     label=label,
                                     color=colors[int(cls)],
                                     line_thickness=3)
                        # print(im0.shape)
                        # print(masks[:, :, i].shape)
                        im0 = plot_one_mask(im0,
                                            color=None,
                                            masks=masks[:, :, i])

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.namedWindow('p', cv2.WINDOW_NORMAL)
                cv2.imshow('p', im0)
                key = cv2.waitKey(0 if pause else 1)
                pause = True if key == ord(' ') else False
                if key == ord('q') or key == ord('e') or key == 27:
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release(
                            )  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                            (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        nargs='+',
        type=str,
        default=
        '/d/projects/research/yolov5/runs/train/silu_s_test_mosaic2/weights/best.pt',
        help='model.pt path(s)')
    parser.add_argument(
        '--source',
        type=str,
        default='/d/projects/research/yolov5/data/balloon/images/val',
        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--name',
                        default='silu_s_test_mosaic', 
                        help='save results to project/name')
    parser.add_argument('--img-size',
                        type=int,
                        default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.5,
                        help='object confidence threshold')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.45,
                        help='IOU threshold for NMS')
    parser.add_argument('--device',
                        default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img',
                        action='store_true',
                        help='display results')
    parser.add_argument('--save-txt',
                        action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf',
                        action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave',
                        default=False,
                        action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes',
                        nargs='+',
                        type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms',
                        action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment',
                        action='store_true',
                        help='augmented inference')
    parser.add_argument('--update',
                        action='store_true',
                        help='update all models')
    parser.add_argument('--project',
                        default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in [
                    'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt'
            ]:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
