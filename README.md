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

