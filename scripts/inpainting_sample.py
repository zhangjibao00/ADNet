import os, ntpath
from util import util
import argparse
import os
from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import BUSIDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from dataloader.data_loader import dataloader
import torchvision.transforms as transforms
from torchsummary import summary
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)



def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img



def save_results(img_paths, save_data, out_dir, score=None, data_name='none'):
    """Save the training or testing results to disk"""

    for i in range(save_data.size(0)):
        print('process image ...... %s' % img_paths[i])#打印当前正在处理的图像路径。这有助于跟踪处理进度
        short_path = ntpath.basename(img_paths[i])  # get image path 提取文件名部分
        name = os.path.splitext(short_path)[0]#获取文件名（不包括扩展名）。os.path.splitext 将文件名分为两部分：主文件名和扩展名。这行代码提取主文件名
        if type(score) == type(None):
            img_name = '%s_%s.png' % (name, data_name)#生成不包含评分的文件名
        else:
            # d_score = score[i].mean()
            # img_name = '%s_%s_%s.png' % (name, data_name, str(round(d_score.item(), 3)))
            img_name = '%s_%s_%s.png' % (name, data_name, str(score))
        # save predicted image with discriminator score
        util.mkdir(out_dir)#确保输出目录 out_dir 存在。如果目录不存在，则创建它。util.mkdir 是一个工具函数，用于创建目录。
        img_path = os.path.join(out_dir, img_name)
        img_numpy = util.tensor2im(save_data[i].data)#将张量数据 save_data[i].data 转换为 NumPy 数组
        util.save_image(img_numpy, img_path)#将 NumPy 数组 img_numpy 保存到 img_path 指定的位置

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)


    datal = dataloader(args)

    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []


    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    for _ in range(len(data)):
        batch = next(data)  #should return an image from the dataloader "data"

        input = batch
        image_paths = input['img_path']
        img = batch['img'].to(dist_util.dev())
        mask = batch['mask'].to(dist_util.dev())
        img_s0 = batch['structure'].to(dist_util.dev())
        img_truth = img * 2 - 1
        img_m = mask * img_truth
        img_s = img_s0 * 2 - 1

        c = th.randn_like(img)

        img = th.cat((c, img_m, img_s, mask), dim=1)

        save_results(image_paths, img_m, args.out_dir, data_name='img_m')

        #img = th.cat((b, c), dim=1)     #add a noise channel$
        # if args.data_name == 'BUSI':
        #     slice_ID=path[0].split("_")[-1].split('.')[0]
        # elif args.data_name == 'BRATS':
        #     # slice_ID=path[0].split("_")[2] + "_" + path[0].split("_")[4]
        #     slice_ID=path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, x0 = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample
            save_results(image_paths, sample, args.out_dir, data_name='out')
            save_results(image_paths, x0, args.out_dir, data_name='x0')
           #str(slice_ID)+
def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=2,
        batch_size=1,
        use_ddim=False,
        model_path=r"E:\zhangjibao\Diffusion\scripts\results\savedmodel043000.pt",         #path to pretrain model
        num_ensemble=2,      #number of samples in the ensemble
        gpu_dev = "0,1",
        out_dir='./test/',
        multi_gpu = None, #"0,1,2"
        debug = False,

        mask_type = [3],
        img_file = r"C:\Users\Administrator\Desktop\1",
        mask_file = r"C:\Users\Administrator\Desktop\2",
        structure_file = r'C:\Users\Administrator\Desktop\3',
        loadSize = [256, 256],
        fineSize = [256, 256],
        isTrain = True,
        resize_or_crop = 'resize_and_crop',
        no_augment = True,
        no_shuffle = True,
        nThreads = 0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
