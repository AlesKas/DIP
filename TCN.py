import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size : int) -> None:
        super().__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2) -> None:
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride, padding, dilation))
        self.chomp1 = Chomp1d(padding)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride, padding, dilation))
        self.dropout = nn.Dropout(dropout)
        self.chomp2 = Chomp1d(padding)
        
        
        self.net = nn.Sequential(
            self.conv1,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            self.conv2,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)        

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # out = self.conv1(x)
        # out = self.chomp1(out)
        # out = self.relu(out)
        # out = self.dropout(out)
        
        # out = self.conv2(out)
        # out = self.chomp2(out)
        # out = self.relu(out)
        # out = self.dropout(out)
        
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        res = self.relu(out + res)
        return res
    
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2) -> None:
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
            
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout) -> None:
        super().__init__()
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()
        
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        y : torch.Tensor = self.tcn(x)
        return self.linear(y.transpose(1, 2))