import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class subband_interaction(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(subband_interaction, self).__init__()
        """
        Subband Interaction Module
        """

        self.input_linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.PReLU()
        )
        self.mean_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU()
        )
        self.output_linear = nn.Sequential(
            nn.Linear(hidden_size * 2, input_size),
            nn.PReLU()
        )
        self.norm = nn.GroupNorm(1, input_size)

    def forward(self, input):
        """
        input: [B, F, F_s, T]
        """
        B, G, N, T = input.shape

        # Transform
        group_input = input  # [B, F, F_s, T]
        group_input = group_input.permute(0, 3, 1, 2).contiguous().view(-1, N)  # [B * T * F, F_s]
        group_output = self.input_linear(group_input).view(B, T, G, -1)  # [B, T, F, H]

        # Avg pooling
        group_mean = group_output.mean(2).view(B * T, -1)  # [B * T, H]

        # Concate and transform
        group_output = group_output.view(B * T, G, -1)  # [B * T, F, H]
        group_mean = self.mean_linear(group_mean).unsqueeze(1).expand_as(group_output).contiguous()  # [B * T, F, H]
        group_output = torch.cat([group_output, group_mean], 2)  # [B * T, F, 2H]
        group_output = self.output_linear(group_output.view(-1, group_output.shape[-1]))  # [B * T * F, F_s]
        group_output = group_output.view(B, T, G, -1).permute(0, 2, 3, 1).contiguous()  # [B, F, F_s, T]
        group_output = self.norm(group_output.view(B * G, N, T))  # [B * F, F_s, T]
        output = input + group_output.view(input.shape)  # [B, F, F_s, T]

        return output
