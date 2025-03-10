server_ep=1
client_ep=5

lr=0.001
optimizer='adam'
 
major_percent=0.8

# CIFAR10
# dataset='cifar10'
# partition_type='class'
# n_rounds=200
# n_client_data=1000
# num_users=5
# batch_size=128

# MNIST and Fashion-MNIST
dataset='mnist'  # 'fashion-mnist'
partition_type='class'
round=200
num_users=5
n_client_data=1000
batch_size=128


for n_client_data in 1000 2000 3000 4000 5000
do
    python3 -u main.py \
            --dataset=${dataset} \
            --partition_type=${partition_type} \
            --num_users=${num_users} \
            --n_client_data=${n_client_data} \
            --lr=${lr} \
            --major_percent=${major_percent} \
            --server_ep=${server_ep} \
            --client_ep=${client_ep} \
            --bs=${batch_size}
done