import torch.nn.functional as F




def kl_div(args, x,y):
    # 온도 스케일링 적용
    scaled_logits_x = x / args.t
    scaled_logits_y = y / args.t
    
    # 소프트맥스 및 로그 소프트맥스 적용
    prob_x = F.log_softmax(scaled_logits_x, dim=1)  # 서버 로그 소프트맥스
    prob_y = F.softmax(scaled_logits_y, dim=1)     # 클라이언트 소프트맥스
    
    loss = F.kl_div(prob_x, prob_y, reduction='batchmean')
    return loss
