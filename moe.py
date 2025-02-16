import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MoELayer(nn.Module):
    def __init__(self, config, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([ExpertLayer(config) for _ in range(num_experts)])
        self.gate = nn.Linear(config.n_embd, num_experts)
        
    def forward(self, x):
        # Input shape: (batch, seq_len, n_embd)
        original_shape = x.shape
        x = x.view(-1, original_shape[-1])  # Flatten batch and sequence dimensions
        
        # Gate computation
        gates = self.gate(x)  # Shape: (batch*seq_len, num_experts)
        weights, indices = torch.topk(gates, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        # Compute weighted sum of experts
        expert_outputs = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = indices[:, i]
            for j in range(self.num_experts):
                mask = (expert_idx == j)
                if mask.any():
                    expert_inputs = x[mask]
                    expert_outputs[mask] += weights[:, i:i+1][mask] * self.experts[j](expert_inputs)
        
        return expert_outputs.view(original_shape)
