from flax import linen as nn


class ClassificationHead(nn.Module):
    hidden_size: int
    num_labels: int

    dropout_rate: float = 0.5

    def __call__(self, pooled, deterministic=True):
        x = nn.Dropout(self.dropout_rate)(pooled, deterministic)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.tanh(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic)
        logits = nn.Dense(self.num_labels)(x)
        return logits
