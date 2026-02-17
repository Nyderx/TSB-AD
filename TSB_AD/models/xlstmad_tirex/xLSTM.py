import torch.nn as nn
from xlstm.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig

from .tirex_slstm.cell import sLSTMBlockConfig
from .tirex_slstm.layer import sLSTMLayer


class xLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        seq_len,
        dropout=0.0,
    ):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_len = seq_len

        self.input_re_cast_layer = nn.Linear(input_size, hidden_size)

        self.slstm_config = sLSTMBlockConfig(
            embedding_dim=hidden_size,
            num_heads=4,
            ffn_proj_factor=2.6667,
            num_states=4,
            num_gates=4,
        )
        self.slstm_block = sLSTMLayer(
            self.slstm_config,
            backend="torch",
        )

        self.mlstm_config = mLSTMLayerConfig(
            conv1d_kernel_size=4,
            qkv_proj_blocksize=4,
            num_heads=4,
            proj_factor=2.0,
            embedding_dim=hidden_size,
            bias=False,
            dropout=dropout,
            context_length=seq_len,
        )
        self.mlstm_block = mLSTMLayer(
            self.mlstm_config,
        )

        self.reset_parameters()

    def forward(self, embedded_input, hidden_states=None):
        embedded_input = self.input_re_cast_layer(embedded_input)

        if hidden_states is None:
            hidden_states = {
                "slstm_state": None,
                "mlstm_state": {"mlstm_state": None, "conv_state": None},
            }

        output_seq, hidden_state = self.slstm_block(
            embedded_input,
            hidden_states["slstm_state"],
            return_last_state=True,
        )

        hidden_states["slstm_state"] = hidden_state

        mlstm_output = self.mlstm_block(
            output_seq,
        )

        return mlstm_output, hidden_states

    def reset_parameters(self):
        # for block in self.blocks:
        # self.slstm_block.reset_parameters()
        self.mlstm_block.reset_parameters()
        # self.output_layer.reset_parameters()
