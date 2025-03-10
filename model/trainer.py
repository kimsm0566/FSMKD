import torch
from torch import nn
import copy
from utils.sampling import data_loader
from model.loss_function import kl_div
from model.Fed import FedAvg
from model.eval import evaluate_model


loss_func_ce = nn.CrossEntropyLoss()
loss_func_MKD = nn.CrossEntropyLoss()

class Trainer(object):

    def __init__(self, args):
        if args.algorithm == 'FSMKD':
            self.fsmkd_train = train_fsmkd
        else:
            raise ValueError("Unknown training method")

    def train_fsmkd(self, server, client, args):
        return self.fsmkd_train(server, client, args)
    
def train_fsmkd(server, client, args):
    
    server.head.load_state_dict(client.model.head.state_dict())
    server.tail.load_state_dict(client.model.tail.state_dict())
    
    server_optimizer = torch.optim.Adam(server.parameters(), lr=args.lr)
    client_optimizer = torch.optim.Adam(client.model.body.parameters(), lr=args.lr)
    
    for iter in range(args.local_ep):
        train_loader = data_loader(args.dataset,
                                        client.dataset[0],
                                        client.dataset[1],
                                        args.bs)    
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = torch.from_numpy(images).to(args.device)
            labels = torch.from_numpy(labels).to(args.device)
            
            server_optimizer.zero_grad()
            client_optimizer.zero_grad()
            
            # 클라이언트 헤드 : 서버 바디 : 클라이언트 테일
            logit_server_p = server(images)

            # 클라이언트 헤드 : 클라이언트 바디 : 클라이언트 테일
            logit_client_p = client.model(images)
            
            # Cross-Entropy Loss 계산 (정답 레이블 필요)
            server_loss_ce = loss_func_ce(logit_server_p, labels)
            client_loss_ce = loss_func_ce(logit_client_p, labels)
            
            # KL Divergence 계산
            server_loss_MKD = kl_div(args, logit_server_p, logit_client_p)
            client_loss_MKD = kl_div(args, logit_client_p, logit_server_p)
            # print(f"server_loss_MKD: {server_loss_MKD:.2f}, client_loss_MKD: {client_loss_MKD:.2f}")

            server_loss = server_loss_ce + server_loss_MKD
            client_loss = client_loss_ce + client_loss_MKD
            
            server_loss.backward(retain_graph=True)
            client_loss.backward()

            server_optimizer.step()
            client_optimizer.step()
            
        client.model.head.load_state_dict(server.head.state_dict())
        client.model.tail.load_state_dict(server.tail.state_dict())
        

def train(args, clients, server, testdata):
    
    h_w_locals = [0 for i in range(args.num_users)]
    t_w_locals = [0 for i in range(args.num_users)]
    
    trainer = Trainer(args)
    
    list_client_accuracies = []
    list_server_accuracies = []         
    list_client_loss = []
    list_server_loss = [] 
    
    for iter in range(args.round):
        client_accuracies = 0
        server_accuracies = 0
        client_losses = 0
        server_losses = 0
        
        print(f"-------- Round {iter} --------")
        for idx in range(args.num_users):
            client = clients[idx]
            
            trainer.fsmkd_train(server, client, args)

            server_accuracy, client_accuracy, server_loss, client_loss = evaluate_model(server, client, testdata, args)
        
            print(f"[Client {client.idx}] \tserver acc={server_accuracy*100:.2f}% loss={server_loss:.4f}, client acc={client_accuracy*100:.2f}% loss={client_loss:.4f}")
            
            server_accuracies += server_accuracy
            client_accuracies += client_accuracy
            server_losses += server_loss
            client_losses += client_loss
            
            h_w_locals[idx] = copy.deepcopy(client.model.head.state_dict())
            t_w_locals[idx] = copy.deepcopy(client.model.tail.state_dict())
            
        if iter % 50 == 0:
            # update global weights
            w_avg_head = FedAvg(h_w_locals)
            w_avg_tail = FedAvg(t_w_locals)
            print("-------- Fedavg Complete --------")
            # copy weight to net_glob
            for client in clients:
                client.model.head.load_state_dict(w_avg_head)  
                client.model.tail.load_state_dict(w_avg_tail)
        
        print(f"[Total] server acc={server_accuracies*100/args.num_users:.2f}% loss={server_losses/args.num_users:.4f}, client acc={client_accuracies*100/args.num_users:.2f}% loss={client_losses/args.num_users:.4f}")
        list_client_accuracies.append(client_accuracies/args.num_users)
        list_server_accuracies.append(server_accuracies/args.num_users)
        list_client_loss.append(server_losses/args.num_users)
        list_server_loss.append(client_losses/args.num_users)
        
    return list_server_accuracies, list_client_accuracies, list_server_loss, list_client_loss