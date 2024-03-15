"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

from mapping_new import *


def main():
    args = create_argparser().parse_args()
    test_model = False
    bpp = 1
    mode = 'inn'
    dist_util.setup_dist()
    logger.configure()

    def get_secret(root):
        s = []
        for i in os.listdir(root):
            s.append(os.path.join(root, i))
        return s

    def change_noise(s, mode='pn'):
        with open(s, 'r') as f:
            read_data = f.read()
        data = [i for i in read_data]
        data = np.array(data).astype('float32').reshape((3, 128, 128))
        data = torch.Tensor(data)
        noise = torch.randn((3, 128, 128))
        noise = torch.Tensor(np.random.normal(0, 0.05, (3, 128, 128)))
        if mode == 'pn':
            d = 1 - data * 2
            n = d * torch.abs(noise).view(1, 3, 128, 128)
            n -= torch.mean(n)
            n /= torch.var(n)
            print(compare(data.view(1, 3, 128, 128), decode(n, mode=mode), bpp=1))
            return data.view(1, 3, 128, 128), n
        elif mode == 'si':
            p = math.sqrt(2 / math.pi + 1) + math.sqrt(2 / math.pi)
            for i in range(len(noise)):
                for j in range(len(noise[0])):
                    for k in range(len(noise[0][0])):
                        if data[i][j][k] == 0:
                            noise[i][j][k] = 0
                        else:
                            if noise[i][j][k] >= 0:
                                noise[i][j][k] += p
                            else:
                                noise[i][j][k] -= p

            return data.view(1, 3, 128, 128), noise.view(1, 3, 128, 128)
        elif mode == 'inn':
            noise = torch.randn(3, 128, 128)
            n = torch.zeros(3, 128, 128, dtype=torch.float)
            z_flag = 0
            alpha = 0.03
            for i in range(3 * 128 * 128):
                if z_flag >= 3 * 128 * 128 - 1:
                    noise = torch.randn(3, 128, 128)
                    z_flag = 0
                if data[i // (128 * 128)][(i // 128) % 128][i % 128] == 0:
                    while noise[z_flag // (128 * 128)][(z_flag // 128) % 128][z_flag % 128] < -1 * alpha:
                        z_flag += 1
                        if z_flag >= 3 * 128 * 128 - 1:
                            noise = torch.randn(3, 128, 128)
                            z_flag = 0
                    n[i // (128 * 128)][(i // 128) % 128][i % 128] = \
                        noise[z_flag // (128 * 128)][(z_flag // 128) % 128][z_flag % 128]
                    z_flag += 1
                else:
                    while noise[z_flag // (128 * 128)][(z_flag // 128) % 128][z_flag % 128] > alpha:
                        z_flag += 1
                        if z_flag >= 3 * 128 * 128 - 1:
                            noise = torch.randn(3, 128, 128)
                            z_flag = 0
                    n[i // (128 * 128)][(i // 128) % 128][i % 128] = \
                        noise[z_flag // (128 * 128)][(z_flag // 128) % 128][z_flag % 128]
                    z_flag += 1

            return data.view(1, 3, 128, 128), n.detach().view(1, 3, 128, 128)

        elif mode == 'sf':
            noise = torch.randn(6, 128, 128)
            n = torch.zeros(3, 128, 128, dtype=torch.float)
            z_flag = 0
            f_flag = 0
            for i in range(3 * 128 * 128):
                if data[i // (128 * 128)][(i // 128) % 128][i % 128] == 0:
                    while noise[z_flag // (128 * 128)][(z_flag // 128) % 128][z_flag % 128] < 0:
                        z_flag += 1
                    n[i // (128 * 128)][(i // 128) % 128][i % 128] = \
                        noise[z_flag // (128 * 128)][(z_flag // 128) % 128][z_flag % 128]
                    z_flag += 1
                else:
                    while noise[f_flag // (128 * 128)][(f_flag // 128) % 128][f_flag % 128] > 0:
                        f_flag += 1
                    n[i // (128 * 128)][(i // 128) % 128][i % 128] = \
                        noise[f_flag // (128 * 128)][(f_flag // 128) % 128][f_flag % 128]
                    f_flag += 1
            c = decode(n.detach().view(1, 3, 128, 128), mode='sf')
            logger.log("acc={}".format(compare(c, data.view(1, 3, 128, 128), bpp=1)))

            return data.view(1, 3, 128, 128), n.detach().view(1, 3, 128, 128)
        elif mode == 'db':

            d = data * 2 - 1
            noise = torch.Tensor(np.random.normal(0, 0.5, (3, 128, 128)))
            n = d + noise.view(1, 3, 128, 128)
            return data.view(1, 3, 128, 128), n

        elif mode == 'mn':
            bit_ori, matrix = bit2dec_new([1, 3, 128, 128], 1, 0.045, 0.0001, arg())
            bit_ori = [i for i in bit_ori]
            bit_ori = np.array(bit_ori).astype('float32').reshape((1, 3, 128, 128))
            bit_ori = torch.Tensor(bit_ori)
            matrix = torch.Tensor(matrix)
            return bit_ori, matrix

        else:

            idx = torch.randperm(noise.nelement())
            noise = noise.view(-1)[idx].view(noise.size())
            return decode(noise.view(1, 3, 128, 128), mode='pn'), noise.detach().view(1, 3, 128, 128)

    def decode(noise, mode='pn'):
        x0 = torch.zeros_like(noise)
        print(noise)
        if mode == 'si':
            noise = torch.Tensor(noise.detach().cpu().numpy()).to("cuda")
            noise -= torch.mean(noise)
            noise /= torch.var(noise)
            p = math.sqrt(2 / math.pi + 1) + math.sqrt(2 / math.pi)
            for i in range(len(noise)):
                for j in range(len(noise[0])):
                    for k in range(len(noise[0][0])):
                        for l in range(len(noise[0][0][0])):
                            if noise[i][j][k][l] >= p / 2 or noise[i][j][k][l] <= -p / 2:
                                x0[i][j][k][l] = 1
            return x0
        elif mode == 'pn' or mode == 'sf':
            for i in range(len(noise)):
                for j in range(len(noise[0])):
                    for k in range(len(noise[0][0])):
                        for l in range(len(noise[0][0][0])):
                            if noise[i][j][k][l] < 0:
                                x0[i][j][k][l] = 1
            return x0
        elif mode == 'db':
            for i in range(len(noise)):
                for j in range(len(noise[0])):
                    for k in range(len(noise[0][0])):
                        for l in range(len(noise[0][0][0])):
                            if noise[i][j][k][l] <= 0:
                                x0[i][j][k][l] = 1
            return x0
        elif mode == 'inn':
            noise = torch.Tensor(noise.detach().cpu().numpy()).to("cuda")
            noise -= torch.mean(noise)
            for i in range(len(noise)):
                for j in range(len(noise[0])):
                    for k in range(len(noise[0][0])):
                        for l in range(len(noise[0][0][0])):
                            if noise[i][j][k][l] >= 0:
                                x0[i][j][k][l] = 1
            return x0
        elif mode == 'mn':
            size_group = 1
            gap = 0.0001
            noise = torch.Tensor(noise.detach().cpu().numpy()).to("cuda")
            noise = noise.detach().cpu().numpy()
            p_2 = 2 ** (size_group - 2)
            if p_2 < 1:
                p_2 = 1
            for i in range(len(noise)):
                for j in range(len(noise[0])):
                    for k in range(len(noise[0][0])):
                        for l in range(len(noise[0][0][0])):
                            if abs(noise[i][j][k][l]) > 1:
                                if noise[i][j][k][l] > 0:
                                    noise[i][j][k][l] -= 1
                                else:
                                    noise[i][j][k][l] = -(abs(noise[i][j][k][l]) - 1)
                            if -1 < noise[i][j][k][l] < -0.4:  # 第一组
                                ordinal = 0
                                for s in range(p_2):
                                    low = (s * 0.6) / (2 ** (size_group - 2)) - 1 + gap
                                    high = (s + 1) * 0.6 / (2 ** (size_group - 2)) - 1 - gap
                                    if low < noise[i][j][k][l] < high:
                                        ordinal = s
                                        break
                                state = ordinal
                            elif -0.4 < noise[i][j][k][l] < 0.4:  # 第二组
                                ordinal = 0
                                for s in range(2 ** (size_group - 1)):
                                    low = (s * 0.8) / (2 ** (size_group - 1)) - 0.4 + gap
                                    high = (s + 1) * 0.8 / (2 ** (size_group - 1)) - 0.4 - gap
                                    if low < noise[i] < high:
                                        ordinal = s
                                        break
                                state = ordinal + 2 ** (size_group - 2)
                            else:  # 第三组
                                ordinal = 0
                                for s in range(p_2):
                                    low = (s * 0.6) / (2 ** (size_group - 2)) + 0.4 + gap
                                    high = (s + 1) * 0.6 / (2 ** (size_group - 2)) + 0.4 - gap
                                    if low < noise[i][j][k][l] < high:
                                        ordinal = s
                                        break
                                state = ordinal + 3 * (2 ** (size_group - 2))
                            x0[i][j][k][l] = state
            return x0
        else:
            for i in range(len(noise)):
                for j in range(len(noise[0])):
                    for k in range(len(noise[0][0])):
                        for l in range(len(noise[0][0][0])):
                            if noise[i][j][k][l] < 0:
                                x0[i][j][k][l] = 1
            return x0

    def compare(x, datas, bpp):
        sums = torch.sum(x == datas).item()
        return sums / (args.batch_size * 3 * 128 * 128)

    secret = get_secret(r"D:\pyyyy\guided-diffusion-main\noise\3")
    random.shuffle(secret)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
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
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    i = 0
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        if test_model:
            classes = torch.Tensor([80, 80])
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        noises = []
        datas = []
        for j in range(args.batch_size):
            data, n = change_noise(secret[i], mode=mode)
            datas.append(data)
            noises.append(n)
            i += 1
        noises = torch.cat(noises, axis=0).to("cuda")
        n0 = noises.detach()
        datas = torch.cat(datas, axis=0).to("cuda")
        if test_model:
            noises = None
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            noise=noises,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        if not test_model:
            addnoise = model_fn(sample, 0.0, y=classes)
            add, addnoise = th.split(addnoise, 3, dim=1)
            similarity1 = torch.cosine_similarity(n0.contiguous().view(args.batch_size, -1),
                                                  addnoise.contiguous().view(args.batch_size, -1))
            logger.log("Similarity1:{}".format(similarity1))
            similarity2 = torch.cosine_similarity(n0.contiguous().view(args.batch_size, -1),
                                                  add.contiguous().view(args.batch_size, -1))
            logger.log("Similarity2:{}".format(similarity2))
            addnoise = decode(addnoise, mode=mode)
            logger.log("acc={}".format(compare(addnoise, datas, bpp=bpp)))
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
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
