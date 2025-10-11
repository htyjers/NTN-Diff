import numpy
import torch
import numpy as np
from tqdm import tqdm
from ntn.dct_util import dct_2d, idct_2d, low_pass, high_pass
from ntn.ddim_sampler import DDIM_Sampler

class FBS_Sampler(DDIM_Sampler):

    def __init__(self, model, schedule="linear", **kwargs):
        super(FBS_Sampler, self).__init__(model, schedule, **kwargs)

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
                ref_latent, _, _ = self.p_sample_ddim(ref_latent, unconditional_conditioning, ts,
                                                      mask=mask_inpainting, index=index,
                                                      use_original_steps=use_original_steps,
                                                      unconditional_guidance_scale=1.0,
                                                      unconditional_conditioning=None)
                
                ref_latent_dct = dct_2d(ref_latent, norm='ortho')
                x_dec_dct = dct_2d(x_dec, norm='ortho')
                merged_dct = low_pass(ref_latent_dct, threshold) \
                             + high_pass(x_dec_dct, threshold + 1)
                x_dec = idct_2d(merged_dct, norm='ortho')
                x_dec, _, _ = self.p_sample_ddim(x_dec, cond, ts, index=index,
                                                 use_original_steps=use_original_steps,
                                                 unconditional_guidance_scale=7.5,
                                                 unconditional_conditioning=conds)

                x_dec_dct = dct_2d(x_dec, norm='ortho')
                ref_latent_dct1 = dct_2d(ref_latent1, norm='ortho')
                merged_dct = low_pass(ref_latent_dct1, threshold1) + \
                             high_pass(low_pass(x_dec_dct, threshold2), threshold1 + 1) + \
                             high_pass(ref_latent_dct1, threshold2 + 1)
                ref_latent1 = idct_2d(merged_dct, norm='ortho')

                ref_latent1, _, _ = self.p_sample_ddim(ref_latent1, conds1, ts, index=index,
                                                       use_original_steps=use_original_steps,
                                                       unconditional_guidance_scale=1.0,
                                                       unconditional_conditioning=None)



            else:
                ref_latent1 = intermediates[intermediate_steps.index(ts + 1)] * (1 - mask) + ref_latent1 * mask
                ref_latent1, _, _ = self.p_sample_ddim(ref_latent1, cond, ts, index=index,
                                                       use_original_steps=use_original_steps,
                                                       unconditional_guidance_scale=7.5,
                                                       unconditional_conditioning=conds)

            if callback: callback(i)
        return ref_latent1






