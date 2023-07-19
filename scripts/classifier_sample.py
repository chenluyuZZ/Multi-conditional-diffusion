"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import sys
sys.path.append('/home/zeyu/fht/diffusion/guided-diffusion')
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion.data import * 
import torch.nn.functional as F
from torchvision import models
import bisect
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    create_model,
    add_dict_to_argparser,
    puge_in_defaults,
    args_to_dict,
)
th.cuda.set_device(0)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model,diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    


    
    puge_model = create_model(**puge_in_defaults())
    puge_model = th.nn.Sequential(puge_model,th.nn.Conv2d(in_channels=6,out_channels=3,kernel_size=1))
    puge_model.load_state_dict(
                dist_util.load_state_dict(
                    args.puge_model, map_location=dist_util.dev()
                )
            )
    


    classifier_vgg = models.vgg19()
    classifier_vgg = th.nn.Sequential(classifier_vgg,th.nn.Linear(1000,10))
    classifier_vgg = remove_batchnorm_recursive(classifier_vgg)
    classifier_vgg.load_state_dict(th.load(args.classifier_path))
    classifier_vgg.to(dist_util.dev())
    

    if args.classifier_use_fp16:
        puge_model[0].convert_to_fp16()
        puge_model[1].half()
        classifier_vgg.half()
    
    classifier = th.nn.Sequential(puge_model,classifier_vgg)
    classifier.cuda()
    classifier.eval() # 64*64

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():

            x_in = x.detach().requires_grad_(True)
            y_label = th.ones_like(y).to(dist_util.dev())
            reshape_x= F.interpolate(x_in,size=(64,64),mode='bilinear',align_corners=False)
            adorn_x = classifier[0][1](classifier[0][0](reshape_x,t,y))
            logits_v = classifier[1](adorn_x) 
            log_probs_v = F.log_softmax(logits_v, dim=-1)

            selected_v = log_probs_v[range(len(logits_v)), y_label.view(-1)]
            reshape_grad2  = th.autograd.grad(selected_v.sum(), reshape_x)[0] * args.classifier_scale
            grad = F.interpolate(reshape_grad2,size=(256,256),mode='bilinear',align_corners=False)

            return  grad

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}


        classes = th.randint(
            low=0, high=10, size=(args.batch_size,), device=dist_util.dev()
        )


        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")
        
        arr = np.concatenate(all_images, axis=0)
        arr = arr[(len(all_images)-1) * args.batch_size: args.num_samples]
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[(len(all_images)-1) * args.batch_size: args.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{len(all_images)}_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, arr, label_arr)
        

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=True,
        model_path="/home/zeyu/fht/diffusion/guided-diffusion/models/256x256_diffusion_uncond.pt",
        puge_model = '/home/zeyu/fht/diffusion/guided-diffusion/models/trained_model/model000077.pt',
        classifier_path="/home/zeyu/fht/diffusion/guided-diffusion/models/base_class/vgg19_0.99508.pt",
        classifier_scale=30,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    defaults['attention_resolutions'] = '32,16,8'
    defaults['class_cond'] =False
    defaults['diffusion_steps'] =1000
    defaults['image_size'] = 256
    defaults['learn_sigma'] = True
    defaults['noise_schedule'] = 'linear'
    defaults['num_channels'] = 256
    defaults['num_head_channels'] = 64
    defaults['num_res_blocks'] = 2
    defaults['resblock_updown'] = True
    defaults['use_fp16'] = True
    defaults['use_scale_shift_norm'] = True
    defaults['classifier_use_fp16'] =True
    defaults['timestep_respacing'] ='ddim50'





    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
