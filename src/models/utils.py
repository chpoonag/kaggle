"""Utilities for pytorch NN package"""
#pylint: disable=no-member, invalid-name

import torch as th
from torch import nn

import warnings

def identity(x):
    return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class JumpingKnowledge(nn.Module):
    r"""The Jumping Knowledge aggregation module from `Representation Learning on
    Graphs with Jumping Knowledge Networks <https://arxiv.org/abs/1806.03536>`__
    """
    def __init__(self, mode='cat', in_feats=None, num_layers=None):
        super(JumpingKnowledge, self).__init__()
        assert mode in ['cat', 'max', 'lstm'], \
            "Expect mode to be 'cat', or 'max' or 'lstm', got {}".format(mode)
        self.mode = mode

        if mode == 'lstm':
            assert in_feats is not None, 'in_feats is required for lstm mode'
            assert num_layers is not None, 'num_layers is required for lstm mode'
            hidden_size = (num_layers * in_feats) // 2
            self.lstm = nn.LSTM(in_feats, hidden_size, bidirectional=True, batch_first=True)
            self.att = nn.Linear(2 * hidden_size, 1)
        elif mode == 'cat':
            self.in_feats = in_feats
            self.num_layers = num_layers
            if (self.in_feats is None) or (self.num_layers is None):
                self.linear_layer = None    # to be auto setup later
                warnings.warn("The linear_layer will be automatically set later during forward.", UserWarning)
            else:
                self.linear_layer = nn.Linear(
                    self.num_layers*self.in_feats, 
                    self.in_feats
                )

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters. This comes into effect only for the lstm mode.
        """
        if self.mode == 'lstm':
            self.lstm.reset_parameters()
            self.att.reset_parameters()
        elif self.mode == 'cat':
            self.linear_layer.reset_parameters()


    def forward(self, feat_list):
        r"""

        Description
        -----------
        Aggregate output representations across multiple GNN layers.

        Parameters
        ----------
        feat_list : list[Tensor]
            feat_list[i] is the output representations of a GNN layer.

        Returns
        -------
        Tensor
            The aggregated representations.
        """
        if self.mode == 'cat':
            if self.linear_layer is None:
                feat_last_dims = [i.shape[-1] for i in feat_list]
                assert th.tensor([i==feat_last_dims[0] for i in feat_last_dims]).all(), \
                f"Expect the last dim to be the same for all tensors in feats_list, got {feat_last_dims}."
                self.in_feats = feat_list[0].shape[-1]
                self.num_layers = len(feat_list)
                device = feat_list[0].device
                self.linear_layer = nn.Linear(self.in_feats*self.num_layers, self.in_feats).to(device)
                warnings.warn("The linear_layer has been automatically set to {}.".format(self.linear_layer), UserWarning)
            return self.linear_layer(th.cat(feat_list, dim=-1))
        elif self.mode == 'max':
            return th.stack(feat_list, dim=-1).max(dim=-1)[0]
        else:
            # LSTM
            stacked_feat_list = th.stack(feat_list, dim=1) # (N, num_layers, in_feats)
            alpha, _ = self.lstm(stacked_feat_list)
            alpha = self.att(alpha).squeeze(-1)            # (N, num_layers)
            alpha = th.softmax(alpha, dim=-1)
            return (stacked_feat_list * alpha.unsqueeze(-1)).sum(dim=1)