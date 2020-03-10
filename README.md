IMAGESIZE=640
When I train on my own dataset, whose picture size is 1920*1080, is IMAGESIZE=1920?

In the paper:
the proposed algorithm consequently maintains the real-time detection speed
of over 42 fps with an input resolution of 512 × 512 despite
the significant improvements in performance


after resize, the picture is 416 by 416 (pad)



train the model using gaussian_yolov3_coco.pth as checkpoint:

Using my own dataset

Traceback (most recent call last):
  File "train.py", line 262, in <module>
    main()
  File "train.py", line 128, in main
    model.load_state_dict(state)
  File "/home/docker/.pyenv/versions/3.6.8/lib/python3.6/site-packages/torch/nn/modules/module.py", line 769, in load_state_dict
    self.__class__.__name__, "\n\t".join(error_msgs)))
RuntimeError: Error(s) in loading state_dict for YOLOv3:
	size mismatch for module_list.14.conv.weight: copying a param with shape torch.Size([267, 1024, 1, 1]) from checkpoint, the shape in current model is torch.Size([30, 1024, 1, 1]).
	size mismatch for module_list.14.conv.bias: copying a param with shape torch.Size([267]) from checkpoint, the shape in current model is torch.Size([30]).
	size mismatch for module_list.22.conv.weight: copying a param with shape torch.Size([267, 512, 1, 1]) from checkpoint, the shape in current model is torch.Size([30, 512, 1, 1]).
	size mismatch for module_list.22.conv.bias: copying a param with shape torch.Size([267]) from checkpoint, the shape in current model is torch.Size([30]).
	size mismatch for module_list.28.conv.weight: copying a param with shape torch.Size([267, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([30, 256, 1, 1]).
	size mismatch for module_list.28.conv.bias: copying a param with shape torch.Size([267]) from checkpoint, the shape in current model is torch.Size([30]).

Using original COCO：

nohup: ignoring input
train.py:74: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
  cfg = yaml.load(f)
0310
02:41:58
Setting Arguments.. :  Namespace(cfg='config/gaussian_yolov3_default.cfg', checkpoint='gaussian_yolov3_coco.pth', checkpoint_dir='/data/checkpoints', checkpoint_interval=1000, debug=False, eval_interval=4000, n_cpu=0, tfboard_dir='data/log', use_cuda=True, weights_path=None)
successfully loaded config file:  {'MODEL': {'TYPE': 'YOLOv3', 'BACKBONE': 'darknet53', 'ANCHORS': [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]], 'ANCH_MASK': [[6, 7, 8], [3, 4, 5], [0, 1, 2]], 'N_CLASSES': 1, 'GAUSSIAN': True}, 'TRAIN': {'LR': 0.001, 'MOMENTUM': 0.9, 'DECAY': 0.0005, 'BURN_IN': 1000, 'MAXITER': 500000, 'STEPS': '(400000, 450000)', 'BATCHSIZE': 8, 'SUBDIVISION': 16, 'IMGSIZE': 608, 'LOSSTYPE': 'l2', 'IGNORETHRE': 0.7, 'GRADIENT_CLIP': 2000.0}, 'AUGMENTATION': {'RANDRESIZE': True, 'JITTER': 0.3, 'RANDOM_PLACING': True, 'HUE': 0.1, 'SATURATION': 1.5, 'EXPOSURE': 1.5, 'LRFLIP': True, 'RANDOM_DISTORT': True}, 'TEST': {'CONFTHRE': 0.8, 'NMSTHRE': 0.45, 'IMGSIZE': 416}, 'NUM_GPUS': 1}
!!!!!!!effective_batch_size = batch_size * iter_size = 8 * 16
Gaussian YOLOv3
Gaussian YOLOv3
Gaussian YOLOv3
loading pytorch ckpt... gaussian_yolov3_coco.pth
Traceback (most recent call last):
  File "train.py", line 262, in <module>
    main()
  File "train.py", line 128, in main
    model.load_state_dict(state)
  File "/home/docker/.pyenv/versions/3.6.8/lib/python3.6/site-packages/torch/nn/modules/module.py", line 769, in load_state_dict
    self.__class__.__name__, "\n\t".join(error_msgs)))
RuntimeError: Error(s) in loading state_dict for YOLOv3:
	size mismatch for module_list.14.conv.weight: copying a param with shape torch.Size([267, 1024, 1, 1]) from checkpoint, the shape in current model is torch.Size([30, 1024, 1, 1]).
	size mismatch for module_list.14.conv.bias: copying a param with shape torch.Size([267]) from checkpoint, the shape in current model is torch.Size([30]).
	size mismatch for module_list.22.conv.weight: copying a param with shape torch.Size([267, 512, 1, 1]) from checkpoint, the shape in current model is torch.Size([30, 512, 1, 1]).
	size mismatch for module_list.22.conv.bias: copying a param with shape torch.Size([267]) from checkpoint, the shape in current model is torch.Size([30]).
	size mismatch for module_list.28.conv.weight: copying a param with shape torch.Size([267, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([30, 256, 1, 1]).
	size mismatch for module_list.28.conv.bias: copying a param with shape torch.Size([267]) from checkpoint, the shape in current model is torch.Size([30]).




The reason is that the gaussian_yolov3_coco has some parameter about sigma, which darknet53.conv.74 does not have


--checkpoint gaussian_yolov3_coco.pth



这些阈值怎么确定?
Detection threshold
detect_thresh = 0.3


# Gaussian YOLOv3 in PyTorch
PyTorch implementation of [Gaussian YOLOv3](https://arxiv.org/abs/1904.04620)

如果你在中国， add these two lines in the dockerfile:

RUN pip3 config set global.index-url http://mirrors.aliyun.com/pypi/simple

RUN pip3 config set install.trusted-host mirrors.aliyun.com

修改了ubuntu的~/.pip/pip.conf，换了pip的源，到了docker里面好像不起作用，还是国外源


my notes from the issues
1）.https://github.com/DeNA/PyTorch_YOLOv3/issues/44
'Batchsize affects the final AP, which is the case for both our repo and the original darknet repo.'


2）
1 iteration = 16 batches
1 batch = 4 images
So 100,000 iter = 1,600,000 batches = 6,400,000 images. In trainvalno5k.part there are 117,264 images, so 6,400,000 / 117,264 ~ 54.6 dataset epoches
We use train2017 split of COCO, which has more than 130,000 images.
For 500k iterations at README it takes (500k / 19.4 (iter/min) ) = 430 hours if you use V100.

3）
Let me share the results we know so far :
COCO AP 50:95 is around 0.275 after the lr drop at 200k iter with random resizing, random placing and LR flip (ref: 0.302 with our repo and author's weight)
random placing and / or LR flip improves the AP by 3-4% before the lr drop
still unclear if random distortion works or not
increasing batch size improves the results. The author's batch-size condition for paper results is unclear

