import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime



formatted_time = datetime.now().strftime("%m-%d_%H:%M")  # 예: 2025-03-04 22:38:00

def plot(args, list_server_accuracies, list_client_accuracies, list_server_loss, list_client_loss, clients):


    result_path = os.path.join(args.result_path,
                            args.dataset,
                            f'num_users_{args.num_users}',
                            f'data_partition_{args.partition_type}',
                            f'n_client_data_{args.n_client_data}',
                            f'batch_size_{args.bs}',
                            f'lr_{args.lr}',
                            f'major_percent_{args.major_percent}',
                            f'server_ep_{args.server_ep}',
                            f'client_ep_{args.client_ep}',
                            f'rounds_{args.round}',
                            )

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        
        
    # # 로컬 & 글로벌 테스트 손실 그래프
    plt.figure(figsize=(8, 5))
    plt.plot(list_server_loss, '-.', label='Local Test Loss', color='blue')
    plt.plot(list_client_loss, '-.', label='Global Test Loss', color='red')
    plt.title('Test Losses Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    fig_file = os.path.join(result_path,
                        'loss.png')
    plt.savefig(fig_file, bbox_inches='tight')
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
    fig_file = os.path.join(result_path,
                        'acc.png')
    plt.savefig(fig_file, bbox_inches='tight')
    plt.show()
    
    # # Define a list of colors for different clients
    # colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    # plt.figure(figsize=(8, 5))
    # for idx, client in enumerate(clients):
        
    #     # Assign colors dynamically based on client index
    #     acc_color = colors[idx % len(colors)]  # Cycle through the color list
    #     loss_color = colors[(idx + 1) % len(colors)]  # Use a different color for loss
        
    #     # Plot accuracy
    #     plt.plot(client.acc, '-.', label=f'Client {idx+1} Accuracy', color=acc_color)
        
    #     # Plot loss
    #     plt.plot(client.loss, '-.', label=f'Client {idx+1} Loss', color=loss_color)
        
    #     plt.title(f'Client {idx+1} Accuracy and Loss Over Time')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Metrics')
    #     plt.legend()
    #     plt.grid(True)
        
    #     # Show the plot
    # fig_file = os.path.join(result_path,
    #                     'acc.png')
    # plt.savefig(fig_file, bbox_inches='tight')