import torch.nn as nn
import torch
from model.VGG import VGG_client
from model.VIT import VIT_client


class Client(nn.Module):
    def __init__(self, data, args, idx):
        super(Client, self).__init__()
        self.idx = idx
        self.dataset = data
        self.model = VGG_client(args)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lr)
        
        if args.model == 'vgg':
            self.model = VGG_client(args)
        elif args.model == 'vit':
            self.model = VIT_client(args)