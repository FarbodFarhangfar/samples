import torch.nn as nn


class hidden_unit(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(hidden_unit, self).__init__()
        self.activation = activation
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out


# Here I use the linear model as the network
class learning(nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels=4, unit=hidden_unit, activation=F.relu):
        super(learning, self).__init__()
        assert type(hidden_layers) is list
        self.in_channels = in_channels
        self.hidden_units = nn.ModuleList()
        prev_layer = in_channels
        for hidden in hidden_layers:
            self.hidden_units.append(unit(prev_layer, hidden, activation))
            prev_layer = hidden
        self.final_unit = nn.Linear(prev_layer, out_channels)

    def forward(self, x):
        out = x.view(-1, self.in_channels).float()
        for unit in self.hidden_units:
            out = unit(out)
        out = self.final_unit(out)
        return out