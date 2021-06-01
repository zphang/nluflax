from flax import linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from flax import traverse_util
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.lib import xla_bridge
import torch
import datasets
from typing import Any
import tqdm.auto as tqdm
import math
from dataclasses import dataclass
import transformers


@dataclass
class RoBERTaConfig:
    # Constant across RoBERTa
    max_embeddings_length: int = 514
    num_embeddings: int = 50265
    padding_idx: int = 1
    ln_eps: float = 1e-5
    dropout_rate: float = 0.5
    attention_dropout_rate: float = 0.5

    # Depends on model-type
    hidden_size = 768
    intermediate_size: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    head_size: int = 64

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
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
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


class ClassificationHead(nn.Module):
    hidden_size: int
    num_labels: int

    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, pooled, deterministic=True):
        x = nn.Dropout(self.dropout_rate)(pooled, deterministic)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.tanh(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic)
        logits = nn.Dense(self.num_labels)(x)
        return logits


class RoBERTaClassificationModel(nn.Module):
    config: RoBERTaConfig
    num_labels: int

    @nn.compact
    def __call__(self, input_ids, attention_mask, deterministic=True):
        encoder_outputs = RoBERTaModel(
            config=self.config,
            name="roberta",
        )(input_ids, attention_mask, deterministic)
        logits = ClassificationHead(
            hidden_size=self.config.hidden_size,
            dropout_rate=self.config.dropout_rate,
            num_labels=self.num_labels,
            name="classification_head",
        )(encoder_outputs["pooled"])
        return logits


def onehot(labels, num_classes):
    x = (labels[..., None] == jnp.arange(num_classes)[None])
    return x.astype(jnp.float32)


def cross_entropy_loss(logits, labels):
    log_probs = nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(onehot(labels, num_classes=log_probs.shape[-1]) * log_probs, axis=-1))

def from_frozen(params):
    return {'/'.join(k): v for k, v in traverse_util.flatten_dict(params).items()}

def to_frozen(flat_params):
    return FrozenDict(traverse_util.unflatten_dict({tuple(k.split('/')): v for k, v in flat_params.items()}))

def load_params_from_pt_weights(pt_weights_path, config):
    weights = {k: v.numpy() for k, v in torch.load(pt_weights_path).items()}
    loaded_params = {}
    loaded_params[f"embeddings/embed/embedding"] = weights['roberta.embeddings.word_embeddings.weight']
    loaded_params[f"embeddings/pos/embedding"] = weights['roberta.embeddings.position_embeddings.weight']
    loaded_params[f"embeddings/tok_type/embedding"] = weights['roberta.embeddings.token_type_embeddings.weight']
    loaded_params[f"embeddings/LayerNorm_0/bias"] = weights["roberta.embeddings.LayerNorm.bias"]
    loaded_params[f"embeddings/LayerNorm_0/scale"] = weights["roberta.embeddings.LayerNorm.weight"]
    for i in range(config.num_layers):
        loaded_params[f"layer_{i}/attention/LayerNorm_0/bias"] = weights[f"roberta.encoder.layer.{i}.attention.output.LayerNorm.bias"]
        loaded_params[f"layer_{i}/attention/LayerNorm_0/scale"] = weights[f"roberta.encoder.layer.{i}.attention.output.LayerNorm.weight"]
        loaded_params[f"layer_{i}/attention/self/key/bias"] = weights[f"roberta.encoder.layer.{i}.attention.self.key.bias"].reshape(config.num_heads, config.head_size)
        loaded_params[f"layer_{i}/attention/self/key/kernel"] = weights[f"roberta.encoder.layer.{i}.attention.self.key.weight"].T.reshape(config.hidden_size, config.num_heads, config.head_size)
        loaded_params[f"layer_{i}/attention/self/out/bias"] = weights[f"roberta.encoder.layer.{i}.attention.output.dense.bias"]
        loaded_params[f"layer_{i}/attention/self/out/kernel"] = weights[f"roberta.encoder.layer.{i}.attention.output.dense.weight"].T.reshape(config.num_heads, config.head_size, config.hidden_size)
        loaded_params[f"layer_{i}/attention/self/query/bias"] = weights[f"roberta.encoder.layer.{i}.attention.self.query.bias"].reshape(config.num_heads, config.head_size)
        loaded_params[f"layer_{i}/attention/self/query/kernel"] = weights[f"roberta.encoder.layer.{i}.attention.self.query.weight"].T.reshape(config.hidden_size, config.num_heads, config.head_size)
        loaded_params[f"layer_{i}/attention/self/value/bias"] = weights[f"roberta.encoder.layer.{i}.attention.self.value.bias"].reshape(config.num_heads, config.head_size)
        loaded_params[f"layer_{i}/attention/self/value/kernel"] = weights[f"roberta.encoder.layer.{i}.attention.self.value.weight"].T.reshape(config.hidden_size, config.num_heads, config.head_size)
        loaded_params[f"layer_{i}/intermediate/Dense_0/bias"] = weights[f"roberta.encoder.layer.{i}.intermediate.dense.bias"]
        loaded_params[f"layer_{i}/intermediate/Dense_0/kernel"] = weights[f"roberta.encoder.layer.{i}.intermediate.dense.weight"].T
        loaded_params[f"layer_{i}/output/Dense_0/bias"] = weights[f"roberta.encoder.layer.{i}.output.dense.bias"]
        loaded_params[f"layer_{i}/output/Dense_0/kernel"] = weights[f"roberta.encoder.layer.{i}.output.dense.weight"].T
        loaded_params[f"layer_{i}/output/LayerNorm_0/bias"] = weights[f"roberta.encoder.layer.{i}.output.LayerNorm.bias"]
        loaded_params[f"layer_{i}/output/LayerNorm_0/scale"] = weights[f"roberta.encoder.layer.{i}.output.LayerNorm.weight"]
    loaded_params["BertPooler_0/Dense_0/kernel"] = weights["roberta.pooler.dense.weight"].T
    loaded_params["BertPooler_0/Dense_0/bias"] = weights["roberta.pooler.dense.bias"]
    loaded_params = jax.device_put(to_frozen(loaded_params))
    return loaded_params

def insert_roberta_params(params, roberta_params):
    params = unfreeze(params)
    params["roberta"] = roberta_params
    params = freeze(params)
    return params


def learning_rate_scheduler(lr, total_steps):
    def f(step):
        return lr * (total_steps - step) / total_steps
    return f


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        #'logits': logits,
    }
    return metrics


def convert_batch(raw_batch):
    return {
        "input_ids": np.array(raw_batch["input_ids"]),
        "attention_mask": np.array(raw_batch["attention_mask"]),
        "label": np.array(raw_batch["label"]),
    }


def main():
    task_dataset = datasets.load_dataset("glue", name="rte")
    tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
    max_seq_length = 256
    config = RoBERTaConfig(max_seq_length=max_seq_length)
    batch_size = 16
    learning_rate = 1e-5
    num_epochs = 10

    def tokenize_examples(examples: dict):
        if "sentence1" in examples:
            return tokenizer.batch_encode_plus(
                list(zip(examples["sentence1"], examples["sentence2"])),
                padding="max_length",
                max_length=max_seq_length,
                truncation="longest_first",
            )
        else:
            return tokenizer.batch_encode_plus(
                examples["sentence"],
                padding="max_length",
                max_length=max_seq_length,
                truncation="longest_first",
            )

    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            logits = task_model.apply(
                {"params": params},
                batch["input_ids"],
                batch["attention_mask"],
                False,
            )
            loss = cross_entropy_loss(logits, batch["label"])
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, logits), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = compute_metrics(logits, batch['label'])
        return state, metrics

    @jax.jit
    def eval_step(params, batch):
        logits = task_model.apply(
            {"params": params},
            batch["input_ids"],
            batch["attention_mask"],
            True,
        )
        return logits

    def train_epoch(state, train_ds):
        batch_metrics = []
        num_examples = len(train_ds)
        permuted_idx = np.random.permutation(num_examples)
        for i in tqdm.trange(0, num_examples, batch_size):
            batch_idx = permuted_idx[i:i+batch_size]
            batch = convert_batch(train_ds[batch_idx])
            state, metrics = train_step(state, batch)
            batch_metrics.append(metrics)
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]
        }
        return state, epoch_metrics_np

    def eval_model(params, validation_ds, batch_size):
        num_examples = len(validation_ds)
        all_preds = []
        eval_batch_size = batch_size * 2
        for i in tqdm.trange(0, num_examples, eval_batch_size):
            batch = convert_batch(validation_ds[i:i+eval_batch_size])
            preds = jax.device_get(eval_step(params, batch))
            all_preds.append(preds)
        all_preds = np.concatenate(all_preds, axis=0)
        return all_preds

    tokenized_dataset = task_dataset.map(tokenize_examples, batched=True)
    roberta_params = load_params_from_pt_weights(
        "/home/zp489/scratch/working/2105/28_flax/exp3/v1/pytorch_model.bin",
        config=RoBERTaConfig,
    )
    total_steps = num_epochs * math.ceil(len(tokenized_dataset["train"]) / batch_size)
    rng = jax.random.PRNGKey(3)
    rng, init_rng = jax.random.split(rng)
    task_model = RoBERTaClassificationModel(
        config=config,
        num_labels=2,
    )
    dummy_input_ids = jnp.ones([1, max_seq_length]).astype(jnp.int32)
    dummy_mask = jnp.ones([1, max_seq_length])#.astype(jnp.int32)
    params = task_model.init(init_rng, dummy_input_ids, dummy_mask)['params']
    # Insert pretrained RoBERTa encoder params
    params = insert_roberta_params(params, roberta_params)
    tx = optax.adamw(
        learning_rate_scheduler(learning_rate, total_steps=total_steps)
        #learning_rate
    )
    state = train_state.TrainState.create(
      apply_fn=task_model.apply, params=params, tx=tx,
    )
    epoch_metrics = []
    for epoch_i in tqdm.trange(num_epochs):
        state, metrics = train_epoch(state=state, train_ds=tokenized_dataset["train"])
        epoch_metrics.append(metrics)
        all_preds = eval_model(state.params, tokenized_dataset["validation"])
        print(metrics)
        print(
            epoch_i,
            (all_preds.argmax(-1) == np.array(tokenized_dataset["validation"]["label"])).mean()
        )