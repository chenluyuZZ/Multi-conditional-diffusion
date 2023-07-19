def class_arg():
    return dict(
        image_size = 256,
        in_channels = 3,
        model_channels = 192,
        out_channels = 10,
        num_res_blocks = 2,
        attention_resolutions = [32,16,8],
        dropout = 0.0,
        channel_mult = (1,2,4,8)
    )
import sys
sys.path.append('/home/zeyu/fht/diffusion/guided-diffusion')
from guided_diffusion.unet import EncoderUNetModel_64
import torch as th
model = EncoderUNetModel_64(**class_arg())

x = th.rand(8,3,256,256)
t = th.randint(low=0,high=10,size=(8,))
pred = model(x,t)
