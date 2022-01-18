import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from apex import amp

from mtaf.model.discriminator import get_fc_discriminator
from mtaf.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from mtaf.utils.func import loss_calc, bce_loss
from mtaf.utils.loss import entropy_loss, kl_divergence, mse_loss
from mtaf.utils.func import prob_2_entropy

def train_advent(model, source_loader, target_loader_list, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES

    num_targets = len(target_loader_list)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)
    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux.train()
    d_aux.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.AMP:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=cfg.TRAIN.AMP_OPTIM
        )

    # discriminators' optimizers
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                          betas=(0.9, 0.99))
    if cfg.TRAIN.AMP:
        d_main, optimizer_d_main = amp.initialize(
            d_main, optimizer_d_main, opt_level=cfg.TRAIN.AMP_OPTIM
        )
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    if cfg.TRAIN.AMP:
        d_aux, optimizer_d_aux = amp.initialize(
            d_aux, optimizer_d_aux, opt_level=cfg.TRAIN.AMP_OPTIM
        )

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    source_loader_iter = enumerate(source_loader)
    target_loader_iter_list = [enumerate(target_loader_list[i]) for i in range(num_targets)]
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)
        # adapt LR if needed
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, cfg)

        # reset optimizers
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False


        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        # train on source
        _, batch = source_loader_iter.__next__()

        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)

        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        if cfg.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # adversarial training to fool the discriminators
        # train on target if pseudo-labels
        pred_trg_main_list = []
        pred_trg_aux_list = []
        for i in range(num_targets):
            _, batch = target_loader_iter_list[i].__next__()
            images, _, _, _ = batch
            pred_trg_aux, pred_trg_main = model(images.cuda(device))

            if cfg.TRAIN.MULTI_LEVEL:
                pred_trg_aux = interp_target(pred_trg_aux)
                d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
                loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
            else:
                loss_adv_trg_aux = 0
                loss_adv_trg_target_aux = 0
            pred_trg_main = interp_target(pred_trg_main)

            pred_trg_main_list.append(pred_trg_main)
            pred_trg_aux_list.append(pred_trg_aux)

            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            loss_adv_trg_main = bce_loss(d_out_main, source_label)

            loss = (  cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                    + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)

            if cfg.TRAIN.AMP:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True

        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            for i in range(num_targets):
                d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
                loss_d_aux = bce_loss(d_out_aux, source_label)
                loss_d_aux = loss_d_aux / (1 + num_targets)
                if cfg.TRAIN.AMP:
                    with amp.scale_loss(loss_d_aux, optimizer_d_aux) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()

        for i in range(num_targets):
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
            loss_d_main = bce_loss(d_out_main, source_label)
            loss_d_main = loss_d_main / 2
            if cfg.TRAIN.AMP:
                with amp.scale_loss(loss_d_main, optimizer_d_main) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_d_main.backward()

        # train with target
        for i in range(num_targets):
            if cfg.TRAIN.MULTI_LEVEL:
                pred_trg_aux = pred_trg_aux_list[i].detach()
                d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
                loss_d_aux = bce_loss(d_out_aux, target_label)
                loss_d_aux = loss_d_aux / 2
                if cfg.TRAIN.AMP:
                    with amp.scale_loss(loss_d_aux, optimizer_d_aux) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_d_aux.backward()
            else:
                loss_d_aux = 0
            pred_trg_main = pred_trg_main_list[i].detach()
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            loss_d_main = bce_loss(d_out_main, target_label)
            loss_d_main = loss_d_main / 2
            if cfg.TRAIN.AMP:
                with amp.scale_loss(loss_d_main, optimizer_d_main) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_d_main.backward()

        optimizer.step()
        for i in range(num_targets):
            optimizer_d_aux.step()
            optimizer_d_main.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()


def train_advent_mdis(model, source_loader, target_loader_list, cfg):
    ''' Multi-Dis. UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES

    num_targets = len(target_loader_list)
    assert num_targets > 1, "Multi-Dis. framework expected multiple targets."

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # seg maps, i.e. output, level
    d_main_list = []
    for i in range(num_targets):
        d_main = get_fc_discriminator(num_classes=num_classes)
        d_main.train()
        d_main.to(device)
        d_main_list.append(d_main)
    # feature level
    d_aux_list = []
    for i in range(num_targets):
        d_aux = get_fc_discriminator(num_classes=num_classes)
        d_aux.train()
        d_aux.to(device)
        d_aux_list.append(d_aux)

    if num_targets > 2:
        d_target_list = []
        d_target_aux_list = []
        for i in range(num_targets):
            d_target = get_fc_discriminator(num_classes=num_classes)
            d_target.train()
            d_target.to(device)
            d_target_list.append(d_target)
            # feature level
            d_target_aux = get_fc_discriminator(num_classes=num_classes)
            d_target_aux.train()
            d_target_aux.to(device)
            d_target_aux_list.append(d_target_aux)
    else:
        d_target = get_fc_discriminator(num_classes=num_classes)
        d_target.train()
        d_target.to(device)
        # feature level
        d_target_aux = get_fc_discriminator(num_classes=num_classes)
        d_target_aux.train()
        d_target_aux.to(device)


    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.AMP:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=cfg.TRAIN.AMP_OPTIM
        )

    # discriminators' optimizers
    optimizer_d_main_list = []
    optimizer_d_aux_list = []
    for i in range(num_targets):
        optimizer_d_main = optim.Adam(d_main_list[i].parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                              betas=(0.9, 0.99))
        if cfg.TRAIN.AMP:
            d_main, optimizer_d_main = amp.initialize(
                d_main_list[i], optimizer_d_main, opt_level=cfg.TRAIN.AMP_OPTIM
            )
            d_main_list[i] = d_main
        optimizer_d_main_list.append(optimizer_d_main)
        optimizer_d_aux = optim.Adam(d_aux_list[i].parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                              betas=(0.9, 0.99))
        if cfg.TRAIN.AMP:
            d_aux, optimizer_d_aux = amp.initialize(
                d_aux_list[i], optimizer_d_aux, opt_level=cfg.TRAIN.AMP_OPTIM
            )
            d_aux_list[i] = d_aux
        optimizer_d_aux_list.append(optimizer_d_aux)
    if num_targets > 2:
        optimizer_d_target_list = []
        optimizer_d_target_aux_list = []
        for i in range(num_targets):
            optimizer_d_target = optim.Adam(d_target_list[i].parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))
            if cfg.TRAIN.AMP:
                d_target, optimizer_d_target = amp.initialize(
                    d_target_list[i], optimizer_d_target, opt_level=cfg.TRAIN.AMP_OPTIM
                )
                d_target_list[i] = d_target
            optimizer_d_target_list.append(optimizer_d_target)
            optimizer_d_target_aux = optim.Adam(d_target_aux_list[i].parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))
            if cfg.TRAIN.AMP:
                d_target_aux, optimizer_d_target_aux = amp.initialize(
                    d_target_aux_list[i], optimizer_d_target_aux, opt_level=cfg.TRAIN.AMP_OPTIM
                )
                d_target_aux_list[i] = d_target_aux
            optimizer_d_target_aux_list.append(optimizer_d_target_aux)
    else:
        optimizer_d_target = optim.Adam(d_target.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                              betas=(0.9, 0.99))
        if cfg.TRAIN.AMP:
            d_target, optimizer_d_target = amp.initialize(
                d_target, optimizer_d_target, opt_level=cfg.TRAIN.AMP_OPTIM
            )
        optimizer_d_target_aux = optim.Adam(d_target_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                              betas=(0.9, 0.99))
        if cfg.TRAIN.AMP:
            d_target_aux, optimizer_d_target_aux = amp.initialize(
                d_target_aux, optimizer_d_target_aux, opt_level=cfg.TRAIN.AMP_OPTIM
            )

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    source_loader_iter = enumerate(source_loader)
    target_loader_iter_list = [enumerate(target_loader_list[i]) for i in range(num_targets)]
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):
        for i in range(num_targets):
            optimizer_d_main = optimizer_d_main_list[i]
            optimizer_d_aux = optimizer_d_aux_list[i]
            optimizer_d_aux.zero_grad()
            optimizer_d_main.zero_grad()
            adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
            adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)
        if num_targets > 2:
            for i in range(num_targets):
                optimizer_d_target = optimizer_d_target_list[i]
                optimizer_d_target_aux = optimizer_d_target_aux_list[i]
                optimizer_d_target_aux.zero_grad()
                optimizer_d_target.zero_grad()
                adjust_learning_rate_discriminator(optimizer_d_target_aux, i_iter, cfg)
                adjust_learning_rate_discriminator(optimizer_d_target, i_iter, cfg)
        else:
            optimizer_d_target.zero_grad()
            adjust_learning_rate_discriminator(optimizer_d_target, i_iter, cfg)
            optimizer_d_target_aux.zero_grad()
            adjust_learning_rate_discriminator(optimizer_d_target_aux, i_iter, cfg)

        # adapt LR if needed
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, cfg)

        # reset optimizers
        for i in range(num_targets):
            d_aux = d_aux_list[i]
            d_main = d_main_list[i]
            for param in d_aux.parameters():
                param.requires_grad = False
            for param in d_main.parameters():
                param.requires_grad = False
        if num_targets > 2:
            for i in range(num_targets):
                d_target_aux = d_target_aux_list[i]
                d_target = d_target_list[i]
                for param in d_target_aux.parameters():
                    param.requires_grad = False
                for param in d_target.parameters():
                    param.requires_grad = False
        else:
            for param in d_target.parameters():
                param.requires_grad = False
            for param in d_target_aux.parameters():
                param.requires_grad = False


        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        # train on source
        _, batch = source_loader_iter.__next__()

        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)

        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        if cfg.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # adversarial training to fool the discriminators
        # train on target if pseudo-labels
        pred_trg_main_list = []
        pred_trg_aux_list = []
        for i in range(num_targets):
            d_aux = d_aux_list[i]
            d_main = d_main_list[i]

            _, batch = target_loader_iter_list[i].__next__()
            images, _, _, _ = batch
            pred_trg_aux, pred_trg_main = model(images.cuda(device))

            if cfg.TRAIN.MULTI_LEVEL:
                pred_trg_aux = interp_target(pred_trg_aux)
                d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
                loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
                if num_targets > 2:
                    loss_adv_trg_target_aux = 0
                    for j in range(num_targets):
                        d_target_aux = d_target_aux_list[j]
                        if i == j:
                            continue
                        d_out_target_aux = d_target_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
                        loss_adv_trg_target_aux = loss_adv_trg_target_aux + bce_loss(d_out_target_aux, 0)
                    loss_adv_trg_target_aux = loss_adv_trg_target_aux / (num_targets - 1)
                else:
                    d_out_target_aux = d_target_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
                    loss_adv_trg_target_aux = bce_loss(d_out_target_aux, 1-i)
            else:
                loss_adv_trg_aux = 0
                loss_adv_trg_target_aux = 0
            pred_trg_main = interp_target(pred_trg_main)

            pred_trg_main_list.append(pred_trg_main)
            pred_trg_aux_list.append(pred_trg_aux)

            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            loss_adv_trg_main = bce_loss(d_out_main, source_label)

            if num_targets > 2:
                loss_adv_trg_target = 0
                for j in range(num_targets):
                    d_target = d_target_list[j]
                    if i == j:
                        continue
                    d_out_target = d_target(prob_2_entropy(F.softmax(pred_trg_main)))
                    loss_adv_trg_target = loss_adv_trg_target + bce_loss(d_out_target, 0)
                loss_adv_trg_target = loss_adv_trg_target / (num_targets - 1)
            else:
                d_out_target = d_target(prob_2_entropy(F.softmax(pred_trg_main)))
                loss_adv_trg_target = bce_loss(d_out_target, 1-i)

            loss = (  cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                    + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux
                    + cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_target
                    + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_target_aux)

            if cfg.TRAIN.AMP:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        if num_targets > 2:
            for i in range(num_targets):
                d_target_aux = d_target_aux_list[i]
                d_target = d_target_list[i]
                for param in d_target_aux.parameters():
                    param.requires_grad = True
                for param in d_target.parameters():
                    param.requires_grad = True
        else:
            for param in d_target.parameters():
                param.requires_grad = True
            for param in d_target_aux.parameters():
                param.requires_grad = True
        for i in range(num_targets):
            d_aux = d_aux_list[i]
            d_main = d_main_list[i]
            for param in d_aux.parameters():
                param.requires_grad = True
            for param in d_main.parameters():
                param.requires_grad = True

        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            for i in range(num_targets):
                d_aux = d_aux_list[i]
                d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
                loss_d_aux = bce_loss(d_out_aux, source_label)
                loss_d_aux = loss_d_aux / (1 + num_targets)
                if cfg.TRAIN.AMP:
                    with amp.scale_loss(loss_d_aux, optimizer_d_aux_list[i]) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()

        for i in range(num_targets):
            d_main = d_main_list[i]
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
            loss_d_main = bce_loss(d_out_main, source_label)
            loss_d_main = loss_d_main / 2
            if cfg.TRAIN.AMP:
                with amp.scale_loss(loss_d_main, optimizer_d_main_list[i]) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_d_main.backward()

        # train with target
        for i in range(num_targets):
            d_aux = d_aux_list[i]
            d_main = d_main_list[i]
            if cfg.TRAIN.MULTI_LEVEL:
                pred_trg_aux = pred_trg_aux_list[i].detach()
                d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
                loss_d_aux = bce_loss(d_out_aux, target_label)
                loss_d_aux = loss_d_aux / 2
                if cfg.TRAIN.AMP:
                    with amp.scale_loss(loss_d_aux, optimizer_d_aux_list[i]) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_d_aux.backward()

                if num_targets > 2:
                    for j in range(num_targets):
                        d_out_target_aux = d_target_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
                        if i == j:
                            loss_d_target_aux = bce_loss(d_out_target_aux, 0) / 2
                            if cfg.TRAIN.AMP:
                                with amp.scale_loss(loss_d_target_aux, optimizer_d_target_aux_list[j]) as scaled_loss:
                                    scaled_loss.backward()
                            else:
                                loss_d_target_aux.backward()
                        else:
                            loss_d_target_aux = bce_loss(d_out_target_aux, 1) / (2*(num_targets-1))
                            if cfg.TRAIN.AMP:
                                with amp.scale_loss(loss_d_target_aux, optimizer_d_target_aux_list[j]) as scaled_loss:
                                    scaled_loss.backward()
                            else:
                                loss_d_target_aux.backward()
                else:
                    d_out_target_aux = d_target_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
                    loss_d_target_aux = bce_loss(d_out_target_aux, i)
                    loss_d_target_aux = loss_d_target_aux / 2
                    if cfg.TRAIN.AMP:
                        with amp.scale_loss(loss_d_target_aux, optimizer_d_target_aux) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss_d_target_aux.backward()
            else:
                loss_d_aux = 0
                loss_d_target_aux = 0
            pred_trg_main = pred_trg_main_list[i].detach()
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            loss_d_main = bce_loss(d_out_main, target_label)
            loss_d_main = loss_d_main / 2
            if cfg.TRAIN.AMP:
                with amp.scale_loss(loss_d_main, optimizer_d_main_list[i]) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_d_main.backward()

            if num_targets > 2:
                loss_d_target_list = [0 in range(num_targets)]
                for j in range(num_targets):
                    d_target = d_target_list[j]
                    d_out_target = d_target(prob_2_entropy(F.softmax(pred_trg_main)))
                    if i == j:
                        loss_d_target = bce_loss(d_out_target, source_label) / 2
                        if cfg.TRAIN.AMP:
                            with amp.scale_loss(loss_d_target, optimizer_d_target_list[j]) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss_d_target.backward()
                    else:
                        loss_d_target = bce_loss(d_out_target, target_label) / (2*(num_targets-1))
                        if cfg.TRAIN.AMP:
                            with amp.scale_loss(loss_d_target, optimizer_d_target_list[j]) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss_d_target.backward()
            else:
                d_out_target = d_target(prob_2_entropy(F.softmax(pred_trg_main)))
                loss_d_target = bce_loss(d_out_target, i)
                loss_d_target = loss_d_target / 2
                if cfg.TRAIN.AMP:
                    with amp.scale_loss(loss_d_target, optimizer_d_target) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_d_target.backward()

        optimizer.step()
        for i in range(num_targets):
            optimizer_d_aux = optimizer_d_aux_list[i]
            optimizer_d_main = optimizer_d_main_list[i]
            optimizer_d_aux.step()
            optimizer_d_main.step()

        if num_targets > 2:
            for i in range(num_targets):
                optimizer_d_target_aux = optimizer_d_target_aux_list[i]
                optimizer_d_target = optimizer_d_target_list[i]
                optimizer_d_target_aux.step()
                optimizer_d_target.step()
        else:
            optimizer_d_target.step()
            optimizer_d_target_aux.step()



        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            for i in range(num_targets):
                d_aux = d_aux_list[i]
                d_main = d_main_list[i]
                torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux_{i}.pth')
                torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main_{i}.pth')
            if num_targets > 2:
                for i in range(num_targets):
                    d_target_aux = d_target_aux_list[i]
                    d_target = d_target_list[i]
                    torch.save(d_target_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_target_aux_{i}.pth')
                    torch.save(d_target.state_dict(), snapshot_dir / f'model_{i_iter}_D_target_{i}.pth')
            else:
                torch.save(d_target_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_target_aux.pth')
                torch.save(d_target.state_dict(), snapshot_dir / f'model_{i_iter}_D_target.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()


def train_advent_mtkt(model, source_loader, target_loader_list, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    num_targets = len(target_loader_list)

    assert cfg.TRAIN.MODEL == 'DeepLabv2MTKT', "MTKT framework expected DeepLabv2MTKT model."
    assert num_targets > 1, "MTKT framework expected multiple targets."

    num_classifiers = 1+num_targets

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level
    # seg maps, i.e. output, level
    d_main_list = []
    d_aux_list = []
    for i in range(num_targets):
        d_main = get_fc_discriminator(num_classes=num_classes)
        d_main.train()
        d_main.to(device)
        d_main_list.append(d_main)
        d_aux = get_fc_discriminator(num_classes=num_classes)
        d_aux.train()
        d_aux.to(device)
        d_aux_list.append(d_aux)

    if cfg.TRAIN.TEACHER_LOSS == "MSE":
        teacher_loss = mse_loss
    elif cfg.TRAIN.TEACHER_LOSS == "KL":
        teacher_loss = kl_divergence
    else:
        raise NotImplementedError(f"Not yet supported loss {cfg.TRAIN.TEACHER_LOSS}")



    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.AMP:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=cfg.TRAIN.AMP_OPTIM
        )

    # discriminators' optimizers
    optimizer_d_main_list = []
    optimizer_d_aux_list = []
    for i in range(num_targets):
        optimizer_d_main = optim.Adam(d_main_list[i].parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                              betas=(0.9, 0.99))
        optimizer_d_aux = optim.Adam(d_aux_list[i].parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                              betas=(0.9, 0.99))
        if cfg.TRAIN.AMP:
            d_main, optimizer_d_main = amp.initialize(
                d_main_list[i], optimizer_d_main, opt_level=cfg.TRAIN.AMP_OPTIM
            )
            d_aux, optimizer_d_aux = amp.initialize(
                d_aux_list[i], optimizer_d_aux, opt_level=cfg.TRAIN.AMP_OPTIM
            )
        d_main_list[i] = d_main
        optimizer_d_main_list.append(optimizer_d_main)
        d_aux_list[i] = d_aux
        optimizer_d_aux_list.append(optimizer_d_aux)

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    source_loader_iter = enumerate(source_loader)
    target_loader_iter_list = [enumerate(target_loader_list[i]) for i in range(num_targets)]
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):
        # reset optimizers
        for i in range(num_targets):
            optimizer_d_main = optimizer_d_main_list[i]
            optimizer_d_main.zero_grad()
            adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)
            optimizer_d_aux = optimizer_d_aux_list[i]
            optimizer_d_aux.zero_grad()
            adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for i in range(num_targets):
            for param in d_main_list[i].parameters():
                param.requires_grad = False
            for param in d_aux_list[i].parameters():
                param.requires_grad = False
        # train on source
        _, batch = source_loader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main = model(images_source.cuda(device))
        loss = 0
        loss_seg_src_aux = 0
        if cfg.TRAIN.MULTI_LEVEL:
            for i in range(1, len(pred_src_aux)):
                pred_src_aux[i] = interp(pred_src_aux[i])
                loss_seg_src_aux += loss_calc(pred_src_aux[i], labels, device)
            loss += cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux
        loss_seg_src_main = 0
        for i in range(1, len(pred_src_main)):
            pred_src_main[i] = interp(pred_src_main[i])
            loss_seg_src_main += loss_calc(pred_src_main[i], labels, device)
        loss += cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main

        if cfg.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # adversarial training to fool the discriminators
        # train on target if pseudo-labels
        pred_trg_aux_list = []
        pred_trg_aux_trg_list = []
        pred_trg_main_list = []
        pred_trg_target_list = []
        for i in range(num_targets):
            d_main = d_main_list[i]
            _, batch = target_loader_iter_list[i].__next__()
            images, labels, _, _ = batch
            all_pred_trg_aux, all_pred_trg_main = model(images.cuda(device))
            if cfg.TRAIN.MULTI_LEVEL:
                pred_trg_aux = interp_target(all_pred_trg_aux[i+1])
                pred_trg_aux_list.append(pred_trg_aux)
                pred_trg_aux_trg = interp_target(all_pred_trg_aux[0])
                pred_trg_aux_trg_list.append(pred_trg_aux_trg)
            pred_trg_main = interp_target(all_pred_trg_main[i+1])
            pred_trg_main_list.append(pred_trg_main)
            pred_trg_target = interp_target(all_pred_trg_main[0])
            pred_trg_target_list.append(pred_trg_target)

            loss = 0
            loss_adv_trg_aux = 0
            if cfg.TRAIN.MULTI_LEVEL:
                d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
                loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
                loss += cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux / 2
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            loss_adv_trg_main = bce_loss(d_out_main, source_label)
            loss += cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main / 2

            loss_div_trg = 0
            loss_div_aux = 0
            if i_iter >= cfg.TRAIN.TEACHER_ITERS:
                pred_trg_main = pred_trg_main.detach()
                loss_div_trg = teacher_loss(pred_trg_target, pred_trg_main)
                loss += cfg.TRAIN.LAMBDA_KL_TARGET * loss_div_trg
                if cfg.TRAIN.MULTI_LEVEL:
                    pred_trg_aux = pred_trg_aux.detach()
                    loss_div_aux = teacher_loss(pred_trg_aux_trg, pred_trg_aux)
                loss += cfg.TRAIN.LAMBDA_SEG_AUX * cfg.TRAIN.LAMBDA_KL_TARGET * loss_div_aux / num_targets
            if cfg.TRAIN.AMP:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for i in range(num_targets):
            for param in d_main_list[i].parameters():
                param.requires_grad = True
            for param in d_aux_list[i].parameters():
                param.requires_grad = True
        # train with source
        for i in range(num_targets):
            pred_src_main[i+1] = pred_src_main[i+1].detach()
            d_main = d_main_list[i]
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main[i+1])))
            loss_d_main = bce_loss(d_out_main, source_label)
            loss_d_main = loss_d_main / 2
            if cfg.TRAIN.AMP:
                with amp.scale_loss(loss_d_main, optimizer_d_main_list[i]) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_d_main.backward()
            if cfg.TRAIN.MULTI_LEVEL:
                pred_src_aux[i+1] = pred_src_aux[i+1].detach()
                d_aux = d_aux_list[i]
                d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux[i+1])))
                loss_d_aux = bce_loss(d_out_aux, source_label)
                loss_d_aux = loss_d_aux / 2
                if cfg.TRAIN.AMP:
                    with amp.scale_loss(loss_d_aux, optimizer_d_aux_list[i]) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_d_aux.backward()

        # train with target
        for i in range(num_targets):
            d_main = d_main_list[i]
            pred_trg_main = pred_trg_main_list[i]
            pred_trg_target = pred_trg_target_list[i]
            pred_trg_main = pred_trg_main.detach() # not sure if useful
            pred_trg_target = pred_trg_target.detach()
            d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
            loss_d_main = bce_loss(d_out_main, target_label)
            loss_d_main = loss_d_main / 2
            if cfg.TRAIN.AMP:
                with amp.scale_loss(loss_d_main, optimizer_d_main_list[i]) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_d_main.backward()
            loss_d_aux = 0
            if cfg.TRAIN.MULTI_LEVEL:
                d_aux = d_aux_list[i]
                pred_trg_aux = pred_trg_aux_list[i]
                pred_trg_aux_trg = pred_trg_aux_trg_list[i]
                pred_trg_aux = pred_trg_aux.detach() # not sure if useful
                pred_trg_aux_trg = pred_trg_aux_trg.detach()
                d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
                loss_d_aux = bce_loss(d_out_aux, target_label)
                loss_d_aux = loss_d_aux / 2
                if cfg.TRAIN.AMP:
                    with amp.scale_loss(loss_d_aux, optimizer_d_aux_list[i]) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_d_aux.backward()
        optimizer.step()
        for i in range(num_targets):
            optimizer_d_main = optimizer_d_main_list[i]
            optimizer_d_main.step()
            optimizer_d_aux = optimizer_d_aux_list[i]
            optimizer_d_aux.step()


        current_losses = {'loss_seg_src_main': loss_seg_src_main,
                          'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_div_trg': loss_div_trg,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_d_main': loss_d_main,
                          'loss_d_aux': loss_d_aux}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def train_domain_adaptation(model, source_loader, target_loader_list, cfg):
    if cfg.TRAIN.MT_FRAMEWORK == 'Baseline':
        train_advent(model, source_loader, target_loader_list, cfg)
    elif cfg.TRAIN.MT_FRAMEWORK == 'MDis':
        train_advent_mdis(model, source_loader, target_loader_list, cfg)
    elif cfg.TRAIN.MT_FRAMEWORK == 'MTKT':
        train_advent_mtkt(model, source_loader, target_loader_list, cfg)
    else:
        raise NotImplementedError(f"Not yet supported framework {cfg.TRAIN.MT_FRAMEWORK}")
