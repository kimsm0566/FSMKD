#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np

def partition_data(train_X, train_y, args):

    idx = np.arange(0, len(train_X))
    np.random.shuffle(idx)

    # Preparation
    if args.partition_type == 'class':
        avail_idx = np.arange(len(train_X))
        train_labels = train_y
    elif args.partition_type == 'iid':
        idx = np.arange(0, len(train_X))
        np.random.shuffle(idx)

    # Get data
    client_data_list = []
    client_major_classes = []
    
    for i in range(args.num_users):

        if args.partition_type == 'class':

            client_major_class = np.random.randint(args.n_class)
            client_major_classes.append(client_major_class)  # major class 저장

            avail_X = train_X[avail_idx]
            avail_labels = train_labels[avail_idx]

            major_mask = avail_labels == client_major_class
            major_idx = np.where(major_mask)[0]
            np.random.shuffle(major_idx)
            
            # major 클래스 데이터 개수 설정
            n_major = int(args.n_client_data * args.major_percent)
            major_idx = major_idx[:n_major]

            # minor 클래스 데이터 개수 설정
            n_minor = args.n_client_data - n_major
            minor_idx = np.where(~major_mask)[0]
            np.random.shuffle(minor_idx)
            minor_idx = minor_idx[:n_minor]

            client_data_idx = np.concatenate((major_idx, minor_idx))
            np.random.shuffle(client_data_idx)

            client_data = avail_X[client_data_idx], avail_labels[client_data_idx]
            client_data_list.append(client_data)

            remaining_idx = set(range(len(avail_idx))) - set(client_data_idx)
            avail_idx = avail_idx[list(remaining_idx)]

        elif args.partition_type == 'class-rep':

            client_major_class = np.random.randint(args.n_class)
            major_mask = train_y == client_major_class
            major_idx = np.where(major_mask)[0]
            np.random.shuffle(major_idx)
            
            major_idx = major_idx[:int(args.n_client_data * args.major_percent)]

            minor_idx = np.where(~major_mask)[0]
            np.random.shuffle(minor_idx)
            minor_idx = minor_idx[:args.n_client_data - int(args.n_client_data * args.major_percent)]

            client_data_idx = np.concatenate((major_idx, minor_idx))
            np.random.shuffle(client_data_idx)
            client_data = train_X[client_data_idx], train_y[client_data_idx]
            client_data_list.append(client_data)

        elif args.partition_type == 'iid':

            client_data_idx = idx[i * args.n_client_data:(i + 1) * args.n_client_data]
            client_data = train_X[client_data_idx], train_y[client_data_idx]
            client_data_list.append(client_data)

    for idx, cls in enumerate(client_major_classes):
        print(f"Client {idx}: Major Class {cls}")
    return client_data_list

CIFAR10_TRAIN_MEAN = np.array((0.4914, 0.4822, 0.4465))[None, :, None, None]
CIFAR10_TRAIN_STD = np.array((0.2470, 0.2435, 0.2616))[None, :, None, None]

def data_loader(dataset, inputs, targets, batch_size, is_train=True):

    def cifar10_norm(x):
        x -= CIFAR10_TRAIN_MEAN
        x /= CIFAR10_TRAIN_STD
        return x

    def no_norm(x):
        return x

    if dataset == 'cifar10':
        norm_func = cifar10_norm
    else:
        norm_func = no_norm

    assert inputs.shape[0] == targets.shape[0]
    n_examples = inputs.shape[0]

    sample_rate = batch_size / n_examples
    num_blocks = int(n_examples / batch_size)
    if is_train:
        for i in range(num_blocks):
            mask = np.random.rand(n_examples) < sample_rate
            if np.sum(mask) != 0:
                yield (norm_func(inputs[mask].astype(np.float32) / 255.),
                       targets[mask])
    else:
        for i in range(num_blocks):
            yield (norm_func(inputs[i * batch_size: (i+1) * batch_size].astype(np.float32) / 255.),
                   targets[i * batch_size: (i+1) * batch_size])
        if num_blocks * batch_size != n_examples:
            yield (norm_func(inputs[num_blocks * batch_size:].astype(np.float32) / 255.),
                   targets[num_blocks * batch_size:])
