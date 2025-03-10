import torch
import numpy as np
from utils.datasets import get_data
from utils.option import args_parser
from utils.sampling import partition_data
from utils.plot import plot
from model.VGG import VGG_server
from model.VIT import VIT_server
from model.client import Client
from model.trainer import train

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    data_path = "/workspace/data/"

    # load dataset and split users
    if args.dataset == 'cifar10':
        args.num_channels = 3
        train_X, train_y, test_X, test_y = get_data(args, data_path)
    elif args.dataset == 'mnist':
        args.num_channels = 1
        train_X, train_y, test_X, test_y = get_data(args, data_path)
    test_data = (test_X, test_y)
    args.n_class = len(np.unique(train_y))
    client_data_list = partition_data(train_X, train_y, args)


    # build model
    clients = []
    if args.model == 'vgg':
        server = VGG_server(args=args).to(args.device)
        for idx in range(args.num_users):
            client = Client(client_data_list[idx], args, idx).to(args.device)
            clients.append(client)
    elif args.model == 'vit':
        server = VIT_server(args).to(args.device)
        for idx in range(args.num_users):
            client = Client(client_data_list[idx], args, idx).to(args.device)
            clients.append(client)

    list_server_accuracies, list_client_accuracies, list_server_loss, list_client_loss = train(args, clients, server, test_data)

    plot(args, list_server_accuracies, list_client_accuracies, list_server_loss, list_client_loss)
