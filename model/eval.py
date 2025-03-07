import torch
from torch import nn

from utils.sampling import data_loader
from model.loss_function import kl_div


def evaluate_model(server, client, data, args):
    
    server.head.load_state_dict(client.model.head.state_dict())
    server.tail.load_state_dict(client.model.tail.state_dict())
    
    server.eval()
    client.model.eval()
    x, y = data

    loader = data_loader(args.dataset, x, y, batch_size=args.bs, is_train=False)

    s_acc = 0.
    c_acc = 0.
    s_total_loss = 0.  # 전체 손실 값 초기화
    c_total_loss = 0.
    loss_func_ce = nn.CrossEntropyLoss()
    batch_count = 0  # 배치 개수를 추적

    for xt, yt in loader:
        xt = torch.tensor(xt, requires_grad=False, dtype=torch.float32).to(args.device)
        yt = torch.tensor(yt, requires_grad=False, dtype=torch.int64).to(args.device)

        logit_server_p = server(xt)
        logit_client_p = client.model(xt)

        # Cross-Entropy Loss 계산 (정답 레이블 필요)
        s_loss_ce = loss_func_ce(logit_server_p, yt)
        c_loss_ce = loss_func_ce(logit_client_p, yt)
        
        # KL Divergence 계산
        s_loss_MKD = kl_div(args, logit_server_p, logit_client_p)
        # KL Divergence 계산
        c_loss_MKD = kl_div(args, logit_client_p, logit_server_p)

        # 최종 손실 계산
        s_loss = s_loss_ce + s_loss_MKD
        c_loss = c_loss_ce + c_loss_MKD
        
        s_total_loss += s_loss.item()      
        c_total_loss += c_loss.item()

        # 정확도 계산
        s_preds_labels = torch.squeeze(torch.max(logit_server_p, 1)[1])  # 예측 클래스
        s_acc += torch.sum(s_preds_labels == yt).item()
                
        c_preds_labels = torch.squeeze(torch.max(logit_client_p, 1)[1])  # 예측 클래스
        c_acc += torch.sum(c_preds_labels == yt).item()
        batch_count += 1            # 배치 개수 증가

    
    s_avg_loss = s_total_loss / batch_count  # 평균 손실 계산
    c_avg_loss = c_total_loss / batch_count  # 평균 손실 계산

    s_accuracy = s_acc / x.shape[0]          # 정확도 계산
    c_accuracy = c_acc / x.shape[0]          # 정확도 계산

    return s_accuracy, c_accuracy, s_avg_loss, c_avg_loss  # 정확도와 평균 손실 반환