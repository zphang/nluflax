from flax import linen as nn
import jax.numpy as jnp
from typing import Any


class RoBERTaConfig:
    # Constant across RoBERTa
    max_embeddings_length: int = 512
    num_embeddings: int = 50265
    padding_idx: int = 1
    ln_eps: float = 1e-5
    dropout_rate: float = 0.5
    attention_dropout_rate: float = 0.5
    kernel_init = nn.initializers.xavier_uniform()
    bias_init = nn.initializers.normal(stddev=1e-6)

    # Depends on model-type
    hidden_size = 768
    intermediate_size: int = 3072
    num_layers: int = 12
    num_heads = 16

    # Run-dependent
    max_seq_length: int = 256
    dtype: Any = jnp.float32


class RoBERTaEmbeddings(nn.Module):
    config: RoBERTaConfig

    @nn.compact
    def __call__(self, input_ids, deterministic=True):
        batch_size = input_ids.shape[0]
        # Token Embeddings
        tok_embeds = nn.Embed(
            num_embeddings=self.config.num_embeddings,
            features=self.config.hidden_size,
            name='embed',
        )(input_ids)

        # Pos Embeddings
        raw_pos_ids = jnp.tile(
            jnp.arange(self.config.max_seq_length), (batch_size, 1)
        ).astype(jnp.int32) + (self.config.padding_idx + 1)
        is_padding = (input_ids == self.config.padding_idx).astype(jnp.int32)
        raw_pos_ids = raw_pos_ids * (1 - is_padding)
        pos_ids = (raw_pos_ids + is_padding * self.config.padding_idx)
        pos_embeds = nn.Embed(
            num_embeddings=self.config.max_embeddings_length,
            features=self.config.hidden_size,
            name='pos',
        )(pos_ids)

        # Token-type Embeddings
        tok_type_ids = jnp.zeros(self.config.max_seq_length).reshape(1, -1).astype(jnp.int32)
        tok_type_embeds = nn.Embed(
            num_embeddings=1,
            features=self.config.hidden_size,
            name='tok_type'
        )(tok_type_ids)

        # Combine, layer-norm, dropout
        embeddings = tok_embeds + pos_embeds + tok_type_embeds
        embeddings = nn.LayerNorm(epsilon=self.config.ln_eps)(embeddings)
        embeddings = nn.Dropout(rate=self.config.dropout_rate)(embeddings, deterministic)
        return embeddings


class BertSelfOutput(nn.Module):
    @nn.compact
    def __call__(self, hidden_states, input_tensor, deterministic=True):
        hidden_states = nn.Dense(self.config.hidden_size)(hidden_states)
        return hidden_states


class BertAttention(nn.Module):
    config: RoBERTaConfig

    @nn.compact
    def __call__(self, hidden_states, attention_mask, deterministic=True):
        att_mask = attention_mask[:, None, None, :]
        att_output = nn.SelfAttention(
            num_heads=self.config.num_heads,
            dtype=self.config.dtype,
            qkv_features=self.config.hidden_size,
            out_features=self.config.hidden_size,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            use_bias=True,
            broadcast_dropout=False,
            dropout_rate=self.config.attention_dropout_rate,
            deterministic=deterministic,
            name="self",
        )(hidden_states, att_mask)
        att_output = nn.Dropout(rate=self.config.dropout_rate)(att_output, deterministic)
        hidden_states = nn.LayerNorm(epsilon=self.config.ln_eps)(att_output + hidden_states)
        return hidden_states


class BertIntermediate(nn.Module):
    config: RoBERTaConfig

    @nn.compact
    def __call__(self, hidden_states):
        hidden_states = nn.Dense(self.config.intermediate_size)(hidden_states)
        hidden_states = nn.gelu(hidden_states, approximate=False)
        return hidden_states


class BertOutput(nn.Module):
    config: RoBERTaConfig

    @nn.compact
    def __call__(self, hidden_states, input_tensor, deterministic=True):
        hidden_states = nn.Dense(self.config.hidden_size)(hidden_states)
        hidden_states = nn.Dropout(rate=self.config.dropout_rate)(hidden_states, deterministic)
        hidden_states = nn.LayerNorm(epsilon=self.config.ln_eps)(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    config: RoBERTaConfig

    @nn.compact
    def __call__(self, hidden_states, attention_mask, deterministic=True):
        self_attention_outputs = BertAttention(
            config=self.config,
            name="attention"
        )(
            hidden_states, attention_mask,
        )
        intermediate_output = BertIntermediate(
            config=self.config,
            name="intermediate",
        )(self_attention_outputs)
        layer_output = BertOutput(
            config=self.config,
            name="output"
        )(intermediate_output, self_attention_outputs)
        return layer_output


class BertPooler(nn.Module):
    config: RoBERTaConfig

    @nn.compact
    def __call__(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = nn.Dense(self.config.hidden_size)(first_token_tensor)
        pooled_output = nn.tanh(pooled_output)
        return pooled_output


class RoBERTaModel(nn.Module):
    config: RoBERTaConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask, deterministic=True):
        embeddings = RoBERTaEmbeddings(
            config=self.config,
            name="embeddings",
        )(input_ids)
        hidden_states = embeddings
        for i in range(self.config.num_layers):
            hidden_states = BertLayer(
                config=self.config,
                name=f"layer_{i}",
            )(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                deterministic=deterministic,
            )
        pooled_output = BertPooler(config=self.config)(hidden_states)
        return {
            "pooled": pooled_output,
            "unpooled": hidden_states,
        }
