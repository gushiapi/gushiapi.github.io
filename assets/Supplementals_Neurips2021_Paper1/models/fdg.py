from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import linklink as link
from models.spike_layer import SpikePool, LIFAct


def adjust_temperature_with_fdg(model, data, target, logger, epsilon=0.1, delta=0.2):
    model.train()
    device = next(model.parameters()).device

    model.zero_grad()
    # first compute FDG
    first_layer = None
    for m in model.modules():
        if isinstance(m, SpikePool):
            first_layer = m
            break

    weight = first_layer.pool.weight.data

    fdg = torch.zeros_like(weight)
    aa, bb, cc, dd = weight.shape
    with torch.no_grad():
        for a in range(aa):
            for b in range(bb):
                for c in range(cc):
                    for d in range(dd):
                        if not hasattr(model, 'conv1'):
                            weight[a, b, c, d] += epsilon
                            output = model(data)
                            loss_p = F.cross_entropy(output, target)
                            weight[a, b, c, d] -= 2 * epsilon
                            output = model(data)
                            loss_n = F.cross_entropy(output, target)
                            fdg[a, b, c, d] = (loss_p - loss_n) / (2 * epsilon)
                            weight[a, b, c, d] += epsilon

    link.allreduce(fdg.data)
    cur_temp = get_temperature(model)
    cosine_sim = 0
    tag = 0
    for i in range(3):
        if i == 0:
            pass
        elif i == 1:
            if cur_temp - delta <= 0.:
                continue
            adjust_temperature(model, -delta)
        else:
            adjust_temperature(model, delta)

        model.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        model.sync_gradients()
        cur_grad = first_layer.pool.weight.grad.data
        cur_cosine_sim = nn.CosineSimilarity(dim=0, eps=1e-6)(cur_grad.flatten(), fdg.flatten())
        if cur_cosine_sim > cosine_sim:
            tag = i
            cosine_sim = cur_cosine_sim

    if tag == 1:
        adjust_temperature(model, -delta)
    if tag == 2:
        adjust_temperature(model, delta)

    new_temp = get_temperature(model)
    logger.info('Temperature adjusted to: {}'.format(new_temp))
    return new_temp


def adjust_temperature(model, delta_temp):
    for m in model.modules():
        if isinstance(m, LIFAct):
            m.temp = m.temp + delta_temp


def get_temperature(model):
    cur_temp = None
    for name, m in model.named_modules():
        if isinstance(m, LIFAct):
            cur_temp = m.temp
            break
    return cur_temp
