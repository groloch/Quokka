import torch
import torch.nn as nn

from . import SinePositionalEmbedding, DecoderLayer, print_trainable_parameters


class QuokkaTransformerModel(nn.Module):
    """
    Decoder-only Transformer model with GQA and Sine positional embeddings.
    """
    def __init__(self, vocab_size, max_seq_len, d_model, num_layers, num_heads, num_groups, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = SinePositionalEmbedding(d_model, max_seq_len)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, num_groups, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.size()
        token_embeds = self.token_embedding(input_ids)  # [bsz, seq_len, d_model]
        
        # Add sine positional embeddings
        pos_embeds = self.pos_embedding(seq_len, input_ids.device)  # [1, seq_len, d_model]
        x = token_embeds + pos_embeds

        for layer in self.layers:
            x = layer(x)

        x = self.final_ln(x)
        logits = self.lm_head(x)
        return logits


class QuokkaMambaModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


class QuokkaHybridModel(nn.Module):
    def __init__(self, *args, **kwargs):
        '''
        
        '''
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass


if __name__ == '__main__':
    test_model = 'transformer'

    if test_model == 'transformer':
        '''
        Transformer config for 175m params
        '''
        model = QuokkaTransformerModel(
            vocab_size=28000,
            max_seq_len=32000,
            d_model=1024,
            num_layers=32,
            num_groups=4,
            num_heads=16,
            dim_feedforward=512,
            dropout=0
        )
        model.to(torch.bfloat16).to('cuda')
        print_trainable_parameters(model)

        input_ids = torch.randint(0, 28000, (2, 8))
        input_ids = input_ids.to('cuda')

        logits = model(input_ids)
        print("Logits shape:", logits.shape)
        while True:
            # infinite loop to monitor vram usage
            pass

    elif test_model == 'mamba':
        pass
    elif test_model == 'hybrid':
        pass
    else:
        pass