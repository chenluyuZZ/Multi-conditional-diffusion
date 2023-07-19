"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os

import blobfile as bf
import torch as th
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torchvision import models
import sys
sys.path.append('/home/zeyu/fht/diffusion/guided-diffusion')

from guided_diffusion import dist_util, logger
from guided_diffusion.data import * 
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.losses import Align_Loss
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model,
    create_classifier_64,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
    create_gaussian_diffusion,
    puge_in_defaults,
    class_arg,
    diffusion_defaults,
    mapping
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict


def main():

    args = create_argparser().parse_args()

    dist_util.setup_dist() # 分布式进程

    logger.configure()

    logger.log("creating model and diffusion...")

    puge_model = create_classifier_64(**class_arg())


    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        # if dist.get_rank() == 0:
        logger.log(
            f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
        )
        puge_model.load_state_dict(
            dist_util.load_state_dict(
                args.resume_checkpoint, map_location=dist_util.dev()
            )
        )
    dist.barrier()
    puge_model.to(dist_util.dev())
    dist_util.sync_params(puge_model.parameters())
    if puge_in_defaults()['use_fp16']:
        puge_model.convert_to_fp16()
    

    diffusion = create_gaussian_diffusion(**diffusion_defaults())


    # model1 = models.vgg19()
    # model1 = nn.Sequential(model1,nn.Linear(1000,10))
    # model1 = remove_batchnorm_recursive(model1)
    # model1.load_state_dict(torch.load('/home/zeyu/fht/diffusion/guided-diffusion/models/base_class/vgg19_0.99508.pt',map_location=dist_util.dev()))
    # model1.half()
    # model1.eval()


    # model2 = models.resnet50()
    # model2 = nn.Sequential(model2,nn.Linear(1000,10))
    # model2.load_state_dict(torch.load('/home/zeyu/fht/diffusion/guided-diffusion/models/base_class/res50_epoch99_0.97054.pt',map_location=dist_util.dev()))
    # model2.half()
    # model2.eval()



    # model1.to(dist_util.dev())
    # model2.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )


    # Needed for creating correct EMAs and fp16 parameters.
    # dist_util.sync_params(model1.parameters())
    # dist_util.sync_params(model2.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=puge_model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=1.0
    )

    puge_model = DDP(
        puge_model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    
    data = data_load(
        data_dir=args.data_dir,
        batch_size=args.batch_size, 
        image_size=args.image_size
    )
    if args.val_data_dir:
        val_data = data_load(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size, 
            image_size=args.image_size
        )
    else:
        val_data = None
    


    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    # loss_fn = Align_Loss(puge_model,model1,model2)
    loss_fn = nn.CrossEntropyLoss()
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        
        for idx,(batch, extra) in enumerate(data_loader):
            logger.logkv('epoch',step)
            logger.logkv("batch", idx)
            logger.logkv(
                "samples",
                (idx) * args.batch_size * dist.get_world_size(),
            )
            
            batch = batch # (-1,1.463)
            labels = extra.to(dist_util.dev())

            batch = batch.to(dist_util.dev())
            # Noisy images
            if args.noised and  prefix=='train':
                t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev()) # t 怎么取的
                batch = diffusion.q_sample(batch, t) # 前向过程加噪
                

            else:
                t = th.zeros(batch.shape[0], dtype=th.float16, device=dist_util.dev())

            for i, (sub_batch, sub_labels, sub_t) in enumerate(
                split_microbatches(args.microbatch, batch, labels, t)
            ):  
                
                # logits = model(sub_batch, timesteps=sub_t)
                # 企鹅 和 松狮狗 类别偏移
                # loss_fn._set_up(sub_batch.to(th.float16),sub_t.to(th.float16),sub_labels.to(th.float16),torch.ones_like(sub_labels,dtype=th.float16))


                # loss,loss1,loss2,loss3 = loss_fn.get_loss()
                pred = puge_model(sub_batch.to(th.float16),sub_t.to(th.float16))
                loss = loss_fn(pred,sub_labels)

                logger.logkv('loss',loss.item())
                # logger.logkv('classfier1 CE loss1',loss1.item())
                # logger.logkv('classfier2 CE loss2',loss2.item())
                # logger.logkv('Gradient alignment loss3',loss3.item())



                if loss.requires_grad:
                    if i == 0:
                        mp_trainer.zero_grad()
                    mp_trainer.backward(loss * len(sub_batch) / len(batch))
            mp_trainer.optimize(opt)
            for param in loss_fn.puge_model.parameters():
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data /= dist.get_world_size()
            
            if not idx % args.log_interval:
                logger.dumpkvs()
            

    for step in range(resume_step,args.iterations):
        logger.logkv("step", step )
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        
        forward_backward_log(data)
        # mp_trainer.optimize(opt)
        # if val_data is not None and not step % args.eval_interval:
        #     with th.no_grad():
        #         with model.no_sync():
        #             model.eval()
        #             forward_backward_log(val_data, prefix="val")
        #             model.train()
        if not step % args.log_interval:
            logger.dumpkvs()
        if (step +1) % args.save_interval==0 and dist.get_rank() == 0:
            logger.log("saving model...")
            save_model(mp_trainer, opt, step )


    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr




def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join('/home/zeyu/fht/diffusion/guided-diffusion/models/trained_model/', f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join('/home/zeyu/fht/diffusion/guided-diffusion/models/trained_model/', f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="~/fht/Data/imagenet-10",
        val_data_dir="",
        noised=True,
        iterations=1000, #epoch
        lr=1e-4,
        weight_decay=0.2,
        anneal_lr=True,
        batch_size=32,
        microbatch=8,
        schedule_sampler="uniform",
        resume_checkpoint=None,
        # opt_resume_checkpoint="/home/zeyu/fht/diffusion/guided-diffusion/models/trained_model/opt000019.pt",
        log_interval=10,
        eval_interval=5,
        save_interval=1,
        nums_gpus = 4
        
    )
    defaults.update(classifier_and_diffusion_defaults())
    defaults['classifier_use_fp16'] = True
    defaults['use_fp16'] =True
    defaults['image_size'] =64
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
            
    main()
