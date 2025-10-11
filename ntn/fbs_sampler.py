import numpy
import torch
import numpy as np
from tqdm import tqdm
from ntn.dct_util import dct_2d, idct_2d, low_pass, high_pass,low_pass_and_shuffle
from ntn.ddim_sampler import DDIM_Sampler

import torch.fft
import torch
import torch.nn.functional as F
import torch
from scipy.ndimage import binary_dilation
class FBS_Sampler(DDIM_Sampler):

    def __init__(self, model, schedule="linear", **kwargs):
        super(FBS_Sampler, self).__init__(model, schedule, **kwargs)

    import numpy as np
    import torch
    from scipy.ndimage import binary_dilation

    def shrink_and_weight(self, mask, iterations, win, initial_weight=1.0):
        """
        从未遮挡区域向遮挡区域逐步收缩，依次赋予递减权重。

        :param mask: 输入二值化掩码，形状为 (1, 1, H, W)，未遮挡区域为 1，遮挡区域为 0，类型为 tensor
        :param iterations: 收缩次数，决定迭代次数
        :param initial_weight: 初始权重值，默认为 1.0
        :return: 加权掩码，形状与输入相同
        """
        # 将 mask 转为 numpy 数组以使用 binary_dilation
        mask_np = mask.cpu().numpy().astype(np.float32)

        # 初始化权重掩码
        weighted_mask_np = np.zeros_like(mask_np, dtype=np.float32)
        weighted_mask_np[mask_np == 1] = 1.0  # 未遮挡区域赋初始权重

        # 当前权重值
        current_weight = initial_weight

        # 初始化膨胀区域（未遮挡区域）
        current_region = mask_np.copy()

        for _ in range(iterations):
            # 计算下一次膨胀区域
            next_region = binary_dilation(current_region, structure=np.ones((1, 1, win, win))).astype(np.float32)

            # 找到新覆盖的遮挡区域
            new_region = (next_region - current_region) * (mask_np == 0)

            # 更新权重掩码，给新覆盖的区域赋值当前权重的减半值
            current_weight /= 2.0
            weighted_mask_np += new_region * current_weight

            # 更新当前区域
            current_region = next_region

            # 如果当前区域没有变化，则提前退出
            if not np.any(new_region):
                break

        weighted_mask1 = torch.tensor(weighted_mask_np, dtype=torch.float32, device=mask.device)
        return weighted_mask1

    @torch.no_grad()
    def decode(self, ref_latent, cond, t_dec, unconditional_guidance_scale,
                         unconditional_conditioning, mask, unmask, conds, conds1, use_original_steps=False, callback=None,
                         threshold=3, end_step=500):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_dec]
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")
        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)

        ref_latent = torch.randn_like(ref_latent)
        x_dec = torch.randn_like(ref_latent)
        ref_latent1 = torch.randn_like(ref_latent)

        total_elements = mask.numel()
        zero_elements = (mask == 0).sum().item()
        zero_ratio = zero_elements / total_elements


        mask_inpainting = mask
        threshold = 90 + 20 * zero_ratio
        threshold1 = 15 - 5 * zero_ratio
        threshold2 = 80 + 20 * zero_ratio


        intermediate_steps = unmask['intermediate_steps']
        intermediates = unmask['intermediates']
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((ref_latent.shape[0],), step, device=ref_latent.device, dtype=torch.long)
            if step >= end_step:

                ref_latent = intermediates[intermediate_steps.index(ts + 1)] * (1 - mask) + ref_latent * mask
                ref_latent, ref_latent0, _ = self.p_sample_ddim(ref_latent, unconditional_conditioning, ts,
                                                      mask=mask_inpainting, index=index,
                                                      use_original_steps=use_original_steps,
                                                      unconditional_guidance_scale=1.0,
                                                      unconditional_conditioning=None)
                ref_latent_dct = dct_2d(ref_latent, norm='ortho')
                x_dec_dct = dct_2d(x_dec, norm='ortho')
                merged_dct = low_pass(ref_latent_dct, threshold) \
                             + high_pass(x_dec_dct, threshold + 1)
                x_dec = idct_2d(merged_dct, norm='ortho')
                x_dec, x_dec0, _ = self.p_sample_ddim(x_dec, cond, ts, index=index,
                                                 use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=7.5,
                                                 unconditional_conditioning=conds)

                x_dec_dct = dct_2d(x_dec, norm='ortho')
                ref_latent_dct1 = dct_2d(ref_latent1, norm='ortho')
                merged_dct = low_pass(ref_latent_dct1, threshold1) + \
                             high_pass(low_pass(x_dec_dct, threshold2), threshold1 + 1) + \
                             high_pass(ref_latent_dct1, threshold2 + 1)
                ref_latent1 = idct_2d(merged_dct, norm='ortho')

                ref_latent1, ref_latent10, _ = self.p_sample_ddim(ref_latent1, conds1, ts, index=index,
                                                       use_original_steps=use_original_steps,
                                                       unconditional_guidance_scale=1.0,
                                                       unconditional_conditioning=None)



            else:
                ref_latent1 = intermediates[intermediate_steps.index(ts + 1)] * (1 - mask) + ref_latent1 * mask
                ref_latent1, ref_latent10, _ = self.p_sample_ddim(ref_latent1, cond, ts, index=index,
                                                       use_original_steps=use_original_steps,
                                                       unconditional_guidance_scale=5.5,
                                                       unconditional_conditioning=conds)

            if callback: callback(i)
        return ref_latent1


