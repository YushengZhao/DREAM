import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, n_layers,
                 dropout=0.5, use_bn=True):
        super(MLP, self).__init__()
        self.use_bn = use_bn
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if n_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(n_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=3, mlp_layers=1, dropout=0.5,
                 train_eps=True):
        super(GIN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.train_eps = train_eps

        self.convs = nn.ModuleList()
        if n_layers == 1:
            self.convs.append(
                GINConv(MLP(in_channels, hidden_channels, out_channels, mlp_layers, dropout), train_eps=train_eps))
        else:
            self.convs.append(
                GINConv(MLP(in_channels, hidden_channels, hidden_channels, mlp_layers, dropout), train_eps=train_eps))
            for layer in range(self.n_layers - 2):
                self.convs.append(GINConv(MLP(hidden_channels, hidden_channels, hidden_channels, mlp_layers, dropout),
                                          train_eps=train_eps))
            self.convs.append(
                GINConv(MLP(hidden_channels, hidden_channels, out_channels, mlp_layers, dropout), train_eps=train_eps))

    def forward(self, x, adj):
        for i in range(self.n_layers - 1):
            x = self.convs[i](x, adj)
            x = F.relu(x)
        x = self.convs[-1](x, adj)
        return x


class GCN(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=5, dropout=0.5, norm_info=None,
                 act='F.relu', input_layer=False, output_layer=False, bias=True, add_self_loops=True):

        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.dropout = dropout
        if norm_info is None:
            norm_info = {'is_norm': False, 'norm_type': 'LayerNorm'}
        self.is_norm = norm_info['is_norm']
        self.norm = eval('nn.' + norm_info['norm_type'])
        self.act = eval(act)
        if input_layer:
            self.input_linear = nn.Linear(in_features=in_channels, out_features=hidden_channels)
        if output_layer:
            self.output_linear = nn.Linear(in_features=hidden_channels, out_features=out_channels)
            self.output_normalization = self.norm_type(hidden_channels)
        self.convs = nn.ModuleList()
        if self.is_norm:
            self.norms = nn.ModuleList()
        else:
            self.norms = None

        for i in range(n_layers):
            if i == 0 and not self.input_layer:
                in_hidden = in_channels
            else:
                in_hidden = hidden_channels
            if i == n_layers - 1 and not self.output_layer:
                out_hidden = out_channels
            else:
                out_hidden = hidden_channels
            self.convs.append(GCNConv(in_hidden, out_hidden, bias=bias, add_self_loops=add_self_loops))
            if self.is_norm:
                self.norms.append(self.norm_type(in_hidden))
        self.convs[-1].last_layer = True

    def forward(self, x, adj):
        if self.input_layer:
            x = self.input_linear(x)
            x = self.input_drop(x)
            x = self.act(x)

        for i, layer in enumerate(self.convs):
            if self.is_norm:
                x_res = self.norms[i](x)
                x_res = layer(x_res, adj)
                x = x + x_res
            else:
                x = layer(x, adj)
            if i < self.n_layers - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.output_layer:
            x = self.output_normalization(x)
            x = self.output_linear(x).squeeze(1)

        return x.squeeze(1)


class EmbedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers,
                 dropout, norm_info=None, act=F.relu, input_layer=False, output_layer=False):
        super().__init__()

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.input_layer = input_layer
        self.output_layer = output_layer

        if isinstance(act, str):
            if act.lower() == 'f.relu':
                self.act_fn = F.relu
            elif act.lower() == 'f.leaky_relu':
                self.act_fn = F.leaky_relu
            elif act.lower() == 'f.elu':
                self.act_fn = F.elu
            elif act.lower() == 'f.tanh':
                self.act_fn = torch.tanh
            elif act.lower() == 'f.sigmoid':
                self.act_fn = torch.sigmoid
            elif act.lower() == 'none' or act == '~':
                self.act_fn = None
            else:
                raise ValueError(f"Unsupported activation function string: {act}")
        elif callable(act):
            self.act_fn = act
        elif act is None:
            self.act_fn = None
        else:
            raise TypeError(f"Activation function must be a string or callable, got {type(act)}")

        current_dim = in_channels
        if input_layer:
            self.input_lin = nn.Linear(in_channels, hidden_channels)
            current_dim = hidden_channels
        else:
            self.input_lin = None

        for i in range(n_layers):
            in_dim = hidden_channels if i > 0 or input_layer else in_channels
            out_dim = hidden_channels if i < n_layers - 1 or output_layer else out_channels
            self.layers.append(GCNConv(in_dim, out_dim))

        if output_layer and n_layers > 0:
            self.output_lin = nn.Linear(hidden_channels, out_channels)
        elif not output_layer and n_layers == 0 and input_layer:
            self.output_lin = nn.Linear(hidden_channels, out_channels)
        elif not output_layer and n_layers == 0 and not input_layer:
            self.output_lin = nn.Linear(in_channels, out_channels)
        else:
            self.output_lin = None

    def forward(self, x, edge_index):
        hidden_representations = []

        if self.input_lin is not None:
            x = self.input_lin(x)
            if self.act_fn is not None: x = self.act_fn(x)
            x = self.dropout(x)
            hidden_representations.append(x)
        else:
            pass

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)

            if i < self.n_layers - 1:
                if self.act_fn is not None: x = self.act_fn(x)
                x = self.dropout(x)

            hidden_representations.append(x)
        if not hidden_representations:
            if self.output_lin:
                output = self.output_lin(x)
                hidden_embeds = x
            else:
                output = x
                hidden_embeds = x
            return output, hidden_embeds

        final_gnn_output = hidden_representations[-1]

        if self.output_lin is not None:
            hidden_embeds = final_gnn_output
            output = self.output_lin(hidden_embeds)
        else:
            output = final_gnn_output
            if len(hidden_representations) >= 2:
                hidden_embeds = hidden_representations[-2]
            else:
                print("Warning: Not enough layers to get distinct hidden embeds when output_layer=False. Using final GNN output.")
                hidden_embeds = final_gnn_output

        return output, hidden_embeds


class SimilarityMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, 1)
        self.act = torch.nn.ReLU()
        self.output_act = torch.nn.Tanh()

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.layer2(x)
        return self.output_act(x)
