# Enhanced Shakespeare model with Mixture of Experts - CPU Optimized Version

# I/O
out_dir = 'out-shakespeare-moe'
eval_interval = 250
eval_iters = 20  # reduced for CPU
log_interval = 1  # more frequent logging

# checkpointing
always_save_checkpoint = False

# wandb logging
wandb_log = False
wandb_project = 'shakespeare-moe'
wandb_run_name = 'moe-gpt'

# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 12  # reduced for CPU
block_size = 64  # reduced context size for CPU

# MoE specific settings
num_experts = 4
top_k = 2
expert_capacity_factor = 1.25
expert_dropout = 0.0  # reduced for stability
routing_temperature = 1.0

# smaller model for CPU
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0  # reduced for CPU training

# training
learning_rate = 1e-3
max_iters = 2000  # reduced for CPU
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.99

# learning rate decay settings
warmup_iters = 100
lr_decay_type = 'cosine'
weight_decay = 0.1

# system
device = 'cpu'
dtype = 'float32'  # stick to float32 for CPU
compile = False

# MoE layer positions
moe_layers = [1, 3]  # reduced number of MoE layers

# Expert balancing
balance_loss_weight = 0.01
capacity_factor = 1.25

# tokenizer
vocab_size = None  # determined from dataset
