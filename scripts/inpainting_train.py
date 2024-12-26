import sys
import argparse
sys.path.append("../")
sys.path.append("./")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import BUSIDataset
from guided_diffusion.custom_dataset_loader import CustomDataset
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

import torch as th
from guided_diffusion.train_util import TrainLoop
# from visdom import Visdom
# viz = Visdom(port=8850)
import torchvision.transforms as transforms

from dataloader.data_loader import dataloader


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    logger.log("creating data loader...")

    datal = dataloader(args)

    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion, netd = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        netd.to(dist_util.dev())
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)


    logger.log("training...")
    TrainLoop(
        model=model,
        netd=netd,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=2e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=1000,
        resume_checkpoint=None, #"/results/pretrainedmodel.pt"
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        multi_gpu=None,
        out_dir='./results/',


        mask_type=[3],
        img_file=None,
        mask_file=None,
        structure_file=None,
        loadSize=[256, 256],
        fineSize=[256, 256],
        isTrain=True,
        resize_or_crop='resize_and_crop',
        no_augment=True,
        no_shuffle=True,
        nThreads=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
