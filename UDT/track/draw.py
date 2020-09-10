from os.path import join, isdir, basename
import os
import argparse
import json
import numpy as np
import torch

import cv2
import time as time
from util import crop_chw, gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox
from net import DCFNet
from eval_otb import eval_auc


class TrackerConfig(object):
    # These are the default hyper-params for DCFNet
    # OTB2013 / AUC(0.665)
    feature_path = 'param.pth'
    crop_sz = 125

    lambda0 = 1e-4
    padding = 2
    output_sigma_factor = 0.1
    interp_factor = 0.01
    num_scale = 3
    scale_step = 1.0275
    scale_factor = scale_step ** (np.arange(num_scale) - num_scale / 2)
    min_scale_factor = 0.2
    max_scale_factor = 5
    scale_penalty = 0.9925
    scale_penalties = scale_penalty ** (np.abs((np.arange(num_scale) - num_scale / 2)))

    feat_sz = 125
    net_input_size = [feat_sz, feat_sz]
    net_average_image = np.array([104, 117, 123]).reshape(-1, 1, 1).astype(np.float32)
    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, net_input_size)
    yf = torch.rfft(torch.Tensor(y).view(1, 1, feat_sz, feat_sz).cuda(), signal_ndim=2)
    cos_window = torch.Tensor(np.outer(np.hanning(feat_sz), np.hanning(feat_sz))).cuda()


# class DCFNetTraker(object):
#     def __init__(self, im, init_rect, config=TrackerConfig(), gpu=True):
#         self.gpu = gpu
#         self.config = config
#         self.net = DCFNet(config)
#         self.net.load_param(config.feature_path)
#         self.net.eval()
#         if gpu:
#             self.net.cuda()

#         # confine results
#         target_pos, target_sz = rect1_2_cxy_wh(init_rect)
#         self.min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
#         self.max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

#         # crop template
#         window_sz = target_sz * (1 + config.padding)
#         bbox = cxy_wh_2_bbox(target_pos, window_sz)
#         patch = crop_chw(im, bbox, self.config.crop_sz)

#         target = patch - config.net_average_image
#         self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())
#         self.target_pos, self.target_sz = target_pos, target_sz
#         self.patch_crop = np.zeros((config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)  # buff

#     def track(self, im):
#         for i in range(self.config.num_scale):  # crop multi-scale search region
#             window_sz = self.target_sz * (self.config.scale_factor[i] * (1 + self.config.padding))
#             bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
#             self.patch_crop[i, :] = crop_chw(im, bbox, self.config.crop_sz)

#         search = self.patch_crop - self.config.net_average_image

#         if self.gpu:
#             response = self.net(torch.Tensor(search).cuda())
#         else:
#             response = self.net(torch.Tensor(search))
#         peak, idx = torch.max(response.view(self.config.num_scale, -1), 1)
#         peak = peak.data.cpu().numpy() * self.config.scale_penalties
#         best_scale = np.argmax(peak)
#         r_max, c_max = np.unravel_index(idx[best_scale], self.config.net_input_size)

#         if r_max > self.config.net_input_size[0] / 2:
#             r_max = r_max - self.config.net_input_size[0]
#         if c_max > self.config.net_input_size[1] / 2:
#             c_max = c_max - self.config.net_input_size[1]
#         window_sz = self.target_sz * (self.config.scale_factor[best_scale] * (1 + self.config.padding))

#         self.target_pos = self.target_pos + np.array([c_max, r_max]) * window_sz / self.config.net_input_size
#         self.target_sz = np.minimum(np.maximum(window_sz / (1 + self.config.padding), self.min_sz), self.max_sz)

#         # model update
#         window_sz = self.target_sz * (1 + self.config.padding)
#         bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
#         patch = crop_chw(im, bbox, self.config.crop_sz)
#         target = patch - self.config.net_average_image
#         self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), lr=self.config.interp_factor)

#         return cxy_wh_2_rect1(self.target_pos, self.target_sz)  # 1-index


if __name__ == '__main__':
    # base dataset path and setting
    parser = argparse.ArgumentParser(description='Test DCFNet on OTB')
    parser.add_argument('--dataset', metavar='SET', default='OTB2015',
                        choices=['OTB2013', 'OTB2015'], help='tune on which dataset')
    parser.add_argument('--ret', default='/home/jaren/ret')
    args = parser.parse_args()


    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    dataset = args.dataset
    base_path = join('dataset', dataset)
    json_path = join('dataset', dataset + '.json')
    annos = json.load(open(json_path, 'r'))
    videos = sorted(annos.keys())

    visualization = True
    test_paths = [os.path.join(args.ret, d) for d in os.listdir(args.ret)
                  if os.path.isdir(os.path.join(args.ret, d))]

    # default parameter and load feature extractor network
    config = TrackerConfig()

    speed = []
    # loop videos
    for video_id, video in enumerate(videos):  # run without resetting
        video_path_name = annos[video]['name']
        init_rect = np.array(annos[video]['init_rect']).astype(np.float)
        image_files = [join(base_path, video_path_name, 'img', im_f) for im_f in annos[video]['image_files']]
        n_images = len(image_files)

        result_dir = join('result', dataset, 'videos', video)
        if not isdir(result_dir): makedirs(result_dir)

        tic = time.time()  # time start

        rets = []
        for test_path in test_paths:
            result_path = join(test_path, video + '.txt')
            with open(result_path, 'r') as f:
                res = [[float(x) for x in line.strip().split(',')]
                       for line in f.readlines()]
            rets.append(res)
        for f in range(1, n_images):  # tryack
            im = cv2.imread(image_files[f])

            rects = [ret[f] for ret in rets]


            if visualization:
                im_show = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                for color, rect in zip(colors, rects):
                    target_pos, target_sz = rect1_2_cxy_wh(rect)
                    cv2.rectangle(im_show, (int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2)),
                                  (int(target_pos[0] + target_sz[0] / 2), int(target_pos[1] + target_sz[1] / 2)),
                                  color, 2)
                cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.CV_AA)
                img_path = join(result_dir, basename(image_files[f]))
                cv2.imwrite(img_path, im_show)
                # cv2.imshow(video, im_show)
                # cv2.waitKey(1)


        toc = time.time() - tic
        fps = n_images / toc
        speed.append(fps)
        print('{:3d} Video: {:12s} Time: {:3.1f}s\tSpeed: {:3.1f}fps'.format(video_id, video, toc, fps))

        # save result
        # test_path = join('result', dataset, 'DCFNet_test')
        # if not isdir(test_path): makedirs(test_path)
        # result_path = join(test_path, video + '.txt')
        # with open(result_path, 'w') as f:
        #     for x in res:
        #         f.write(','.join(['{:.2f}'.format(i) for i in x]) + '\n')

    print('***Total Mean Speed: {:3.1f} (FPS)***'.format(np.mean(speed)))

    eval_auc(dataset, 'DCFNet_test', 0, 1)
