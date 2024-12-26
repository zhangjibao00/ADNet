import copy
import functools
import os
from collections import OrderedDict
import blobfile as bf
import torch as th
import itertools
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from util import util
from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import LossAwareSampler, UniformSampler
# from visdom import Visdom
# viz = Visdom(port=8850)
# loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='loss'))
# grad_window = viz.line(Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(),
#                            opts=dict(xlabel='step', ylabel='amplitude', title='gradient'))

from util.visualizer import Visualizer
visualizer = Visualizer()

from torchvision import models
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', th.tensor(target_real_label))
        self.register_buffer('fake_label', th.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = nn.ReLU()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def __call__(self, prediction, target_is_real, is_disc=False):
        if self.gan_mode in ['lsgan', 'vanilla']:
            labels = (self.real_label if target_is_real else self.fake_label).expand_as(prediction).type_as(prediction)
            loss = self.loss(prediction, labels)
        elif self.gan_mode in ['hinge', 'wgangp']:
            if is_disc:
                if target_is_real:
                    prediction = -prediction
                if self.gan_mode == 'hinge':
                    loss = self.loss(1 + prediction).mean()
                elif self.gan_mode == 'wgangp':
                    loss = prediction.mean()
            else:
                loss = -prediction.mean()
        return loss
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        netd,
        classifier,
        diffusion,
        data,
        dataloader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.net_D = netd
        self.dataloader=dataloader
        self.classifier = classifier
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr

        self.visual_names = ['img_m', 'img_truth', 'model_output', 'cal', 'img_s', 'res_t']
        self.loss_names = ['inp', 'recd', 'recz']

        self.L1loss = th.nn.L1Loss()
        self.L2loss = th.nn.MSELoss()
        self.lossNet = VGG16FeatureExtractor()
        self.lossNet.to(dist_util.dev())
        self.GANloss = GANLoss('lsgan')


        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.total_iteration = 1

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self.optimizer_D = th.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters())),
                                         lr=self.lr, betas=(0.0, 0.999))
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_part_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        i = 0
        data_iter = iter(self.dataloader)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):


            try:
                    batch = next(data_iter)
                    #batch, cond, name = next(data_iter)
                    cond = {}
            except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

            self.run_step(batch, cond)

           
            i += 1
          
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()


    def run_step(self, batch, cond):
#        batch=th.cat((batch, cond), dim=1)

#        cond={}
        self.input = batch
        self.image_paths = self.input['img_path']
        self.img = batch['img'].to(dist_util.dev())
        self.mask = batch['mask'].to(dist_util.dev())
        self.img_s0 = batch['structure'].to(dist_util.dev())
        self.img_truth = self.img * 2 - 1
        self.img_m = self.mask * self.img_truth
        self.img_s = self.img_s0 * 2 - 1
        batch = th.cat((self.img_truth, self.img_m, self.img_s, self.mask), dim=1)


        sample = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return sample

    def forward_backward(self, batch, cond):

        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation,
                self.ddp_model,
                self.classifier,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()

            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()

            # if isinstance(self.schedule_sampler, LossAwareSampler):
            #     self.schedule_sampler.update_with_local_losses(
            #         t, losses1[0]["loss"].detach()
            #     )
            self.model_output, self.cal, self.res_t = losses1[0], losses1[1], losses1[2]


            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

            # optimize the completion network parameters
            self.mp_trainer.zero_grad()
            self.backward_G()

            self.total_iteration += 1
            visualizer.display_current_results(self.get_current_visuals(), self.total_iteration)
            visualizer.plot_current_errors(self.total_iteration, self.get_current_errors())
            for name, param in self.ddp_model.named_parameters():
                if param.grad is None:
                    print(name)
            return self.model_output

    def backward_G(self):
        self._freeze(self.net_D)
        D_fake_g = self.net_D(self.model_output)
        D_real_g = self.net_D(self.img_truth)
        self.loss_ad_l = self.L2loss(D_fake_g, D_real_g) * 10
        self.loss_recd = self.L1loss(self.img_truth, self.model_output) * 60
        self.loss_recz = self.L1loss(self.img_truth, self.cal) * 20

        real_feats2 = self.lossNet(self.img_truth)
        fake_feats2 = self.lossNet(self.model_output)
        comp_feats2 = self.lossNet(self.cal)

        self.loss_style = self.style_loss(real_feats2, fake_feats2) + self.style_loss(real_feats2, comp_feats2)
        self.loss_perc = self.perceptual_loss(real_feats2, fake_feats2) + self.perceptual_loss(real_feats2, comp_feats2)
        self.loss_inp = 7 * self.loss_style + self.loss_perc
        # self.loss_kl_s = self.kl_g_s#.mean() * 20
        # self.loss_kl_t = self.kl_g_t#.mean() * 20
        total_loss = 0
        total_loss = self.loss_recd + self.loss_recz + self.loss_inp + self.loss_ad_l
        self.mp_trainer.backward(total_loss)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # global
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        self._unfreeze(self.net_D)
        self.loss_img_dg = self.backward_D_basic(self.net_D, self.img_truth, self.model_output)

    def _freeze(self, *args):
        """freeze the network for forward process"""
        for module in args:
            if module:
                for p in module.parameters():
                    p.requires_grad = False
    def _unfreeze(self, *args):
        """ unfreeze the network for parameter update"""
        for module in args:
            if module:
                for p in module.parameters():
                    p.requires_grad = True

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)
                if isinstance(value, list):
                    # visual multi-scale ouputs
                    # for i in range(len(value)):
                    #     visual_ret[name + str(i)] = util.tensor2im(value[i].data)
                    visual_ret[name] = util.tensor2im(value[-1].data)
                else:
                    visual_ret[name] = util.tensor2im(value.data)
        return visual_ret

    def get_current_errors(self):
        """Return training loss"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, 'loss_' + name).item()
        return errors_ret

    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = th.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = th.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += th.mean(th.abs(A_style - B_style) / (c * w * h))
        return loss_value

    def perceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += th.mean(th.abs(A_feat - B_feat))
        return loss_value

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"savedmodel{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"optsavedmodel{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)



