import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--round', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--frac', type=int, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--bs', type=int, default=1024, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--t', type=float, default=2, help="T")
    parser.add_argument('--alpha', type=float, default=0.5, help="weight sum ratio")
    
    # model arguments
    parser.add_argument('--model', type=str, choices=['vit', 'vgg'], default='vit', help='model name')
    parser.add_argument("--algorithm",type=str, choices=['FSMKD'],default="FSMKD", help="Algorithm: ['FSMKD'].")
    
    #data arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--partition_type', type=str, default='class', help="class, class-rep, iid")
    parser.add_argument("--n_client_data",type=int, default=3000, help="Number of data points for each client.")
    parser.add_argument("--major_percent", type=float, default=0.8, help="Percentage of majority class for client data partition.")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    
    # other arguments
    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    args = parser.parse_args()
    return args