import torch
from torch import nn


class ExpertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # comes in pre-flattened?
            nn.Linear(784, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )

    def forward(self, input_vector):
        return self.net(input_vector)


class GateNetwork(nn.Module):
    def __init__(self, layer_size, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, num_experts),
            nn.Softmax()
        )

    def forward(self, in_vec):
        return self.gate(in_vec)


class MLENetwork(nn.Module):
    def __init__(self, gate_size, num_experts):
        super().__init__()
        self.gate = GateNetwork(gate_size, num_experts)
        self.experts = [ExpertModel() for i in range(num_experts)]

    def forward(self, input):
        expert_ind = torch.argmax(self.gate(input)).item()
        # Do I need to do anything special here?
        expert = self.experts[expert_ind]
        return expert(input)
