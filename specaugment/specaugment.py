import random

import torch
from torchaudio import transforms

from specaugment.nb_SparseImageWarp import sparse_image_warp


def time_warp(spec, W=2):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]
    device = spec.device

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    assert len(horizontal_line_at_ctr) == spec_len

    point_idx = random.randrange(W, spec_len - W)
    #     point_idx = 415
    point_to_warp = horizontal_line_at_ctr[point_idx]
    assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    #     dist_to_warp = 1
    src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device),
                         torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)


def freq_mask(spec, F=20, num_masks=1, replace_with_zero=True):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if f_zero == f_zero + f:
            return cloned

        mask_end = random.randrange(f_zero, f_zero + f)
        if replace_with_zero:
            cloned[0][f_zero:mask_end] = 0
        else:
            cloned[0][f_zero:mask_end] = cloned.mean()

    return cloned


def time_mask(spec, T=30, num_masks=1, replace_with_zero=True):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if t_zero == t_zero + t:
            return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if replace_with_zero:
            cloned[0][:, t_zero:mask_end] = 0
        else:
            cloned[0][:, t_zero:mask_end] = cloned.mean()
    return cloned
