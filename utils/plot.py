import matplotlib.pyplot as plt
from datetime import datetime



formatted_time = datetime.now().strftime("%m-%d_%H:%M")  # 예: 2025-03-04 22:38:00

def plot(args, list_server_accuracies, list_client_accuracies, list_server_loss, list_client_loss):
    # # 로컬 & 글로벌 테스트 손실 그래프
    plt.figure(figsize=(8, 5))
    plt.plot(list_server_loss, '-.', label='Local Test Loss', color='blue')
    plt.plot(list_client_loss, '-.', label='Global Test Loss', color='red')
    plt.title('Test Losses Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'result/losses_{formatted_time}_{args.partition_type}__clientdata{args.n_client_data}_bs{args.bs}.png')  # 그래프 저장
    plt.show()

    # 로컬 & 글로벌 정확도 그래프
    plt.figure(figsize=(8, 5))
    plt.plot(list_client_accuracies, '-.',label='Local Accuracy', color='skyblue')
    plt.plot(list_server_accuracies, '-.',label='Global Accuracy', color='blue')
    plt.title('Accuracies Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'result/accuracies_{formatted_time}_{args.partition_type}_clientdata{args.n_client_data}_bs{args.bs}.png')  # 그래프 저장
    plt.show()