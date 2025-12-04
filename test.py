import einops
import numpy as np
import random
import torch
from PIL import Image
import os
from pytorch_lightning import seed_everything
from ntn.tools import create_model, load_state_dict
from ntn.fbs_sampler import FBS_Sampler
import torchvision.utils as vutils
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))
print("Available GPUs:", torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected Device:", device)
# resolution of the generated image
H = W = 512

num_samples = 1


import torch
import torch.nn.functional as F


def preprocess_mask(maskt, latent):
    """
    下采样 mask 并移动到 ref_latent 所在设备。

    参数:
    - mask: 原始 mask，可能是 numpy 数组
    - ref_latent: 参考 latent，确定目标大小和设备

    返回:
    - 下采样并转移到正确设备的 mask
    """
    # 如果 mask 是 NumPy 数组，转换为 PyTorch 张量
    if isinstance(maskt, np.ndarray):
        maskt = torch.from_numpy(maskt)

    # 确保 mask 是浮点数类型（插值需要）
    maskt = maskt.float()

    # 获取 ref_latent 的设备和目标大小
    device = latent.device
    target_size = latent.shape[-2:]  # (H, W)

    # 下采样 mask 到 target_size
    mask_resized = F.interpolate(maskt, size=target_size, mode='bilinear', align_corners=True)
    mask_resized = mask_resized.to(device)
    return mask_resized


from safetensors.torch import load_file
model = create_model('./models/model_ldm_v15.yaml').cuda()
state_dict = load_file('./models/Realistic_Vision_V6.0_NV_B1.safetensors')
model.load_state_dict(state_dict, strict=False)

sampler = FBS_Sampler(model)


encode_steps = 1000

# set the total steps of the sampling trajectory
decode_steps = 100

# set the value of lambda (0~1), the larger the lambda_end, the shorter the calibration phase is.
lambda_end = 0.6

# the end step of the calibration phase
end_step = encode_steps * lambda_end

ddim_eta = 0

unconditional_guidance_scale = 7.5

caption = 'XXXXX'
img_path = "XXXXX"
mask_path = "XXXXX"

target_prompt = caption
seed = -1

if seed == -1:
    seed = random.randint(0, 100000000000)
seed_everything(seed)

mask = np.array(Image.open(mask_path).resize((H, W)).convert('L'))
mask = (mask.astype(np.int32) / 255.0)
mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32).cuda()

img = np.array(Image.open(img_path).resize((H, W)))
img = img[..., :3]
img = (img.astype(np.float32) / 127.5) - 1.0
img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None, ...].repeat(num_samples, 1, 1, 1).to(dtype=torch.float32).cuda()  # n, 3, 512, 512

un_cond = {"c_crossattn": [model.get_learned_conditioning(['RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3'] * num_samples)]}


cond = {"c_crossattn": [model.get_learned_conditioning([target_prompt] * num_samples)]}
conds = {"c_crossattn": [model.get_learned_conditioning(['text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated'] * num_samples)]}


conds1 = {"c_crossattn": [model.get_learned_conditioning([''] * num_samples)]}
shape = (4, H // 8, W // 8)

mask_resized = preprocess_mask(mask_tensor, torch.randn(1, 4, 64, 64).cuda()).cuda()
encoder_posterior = model.encode_first_stage(img_tensor)
z = (model.get_first_stage_encoding(encoder_posterior)).detach()


sampler.make_schedule(ddim_num_steps=encode_steps)
latent, out = sampler.encode(x0=z, cond=un_cond, t_enc=encode_steps)
sampler.make_schedule(ddim_num_steps=decode_steps)

x_rec = sampler.decode(ref_latent=latent, cond=cond, t_dec=decode_steps,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=un_cond, mask=mask_resized,
                                                    unmask = out, conds = conds,conds1 = conds1,
                                                    threshold=-1,
                                                    end_step=end_step)
composed_image_path ="enter your path"
img_path ="enter your path"
composed_image = model.decode_first_stage(x_rec)
composed_image = (composed_image + 1) / 2
vutils.save_image(composed_image, composed_image_path, normalize=False)

original_image = (img_tensor + 1) / 2

vutils.save_image(original_image, img_path, normalize=False)


