from typing import Any
import math
import numpy as np
import os
import tqdm.auto as tqdm
import urllib
from dataclasses import dataclass
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from flax import traverse_util
from flax.training import train_state
import optax
import torch
import transformers
import datasets
from absl import flags, app


FLAGS = flags.FLAGS
flags.DEFINE_float(
    'learning_rate',
    default=1e-5, help='The learning rate for the AdamW optimizer.',
)
flags.DEFINE_integer(
    'batch_size',
    default=16, help='Batch size for training.',
)
flags.DEFINE_integer(
    'max_seq_length',
    default=256, help='Maximum sequence length',
)
flags.DEFINE_integer(
    'num_epochs',
    default=5, help='Number of train epochs.',
)
flags.DEFINE_string(
    'model_name',
    default="roberta-base", help='"roberta-base" or "roberta-large"',
)
flags.DEFINE_string(
    'task',
    default=None, help='cola/mnli/mrpc/qnli/qqp/rte/sst/stsb/wnli',
)
flags.DEFINE_string(
    'output_dir',
    default=None, help='Directory to output to',
)


NUM_LABELS_DICT = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst": 2,
    "stsb": 1,
    "wnli": 2,
}
INPUT_FIELDS_DICT = {
    "cola": ["sentence"],
    "mnli": ["premise", "hypothesis"],
    "mrpc": ["sentence1", "sentence2"],
    "qnli": ["question", "sentence"],
    "qqp": ["sentence1", "sentence2"],
    "rte": ["sentence1", "sentence2"],
    "sst": ["sentence"],
    "stsb": ["sentence1", "sentence2"],
    "wnli": ["sentence1", "sentence2"],
}


@dataclass
class RoBERTaConfig:
    # Constant across RoBERTa
    max_embeddings_length: int = 514
    num_embeddings: int = 50265
    padding_idx: int = 1
    ln_eps: float = 1e-5
    dropout_rate: float = 0.5
    attention_dropout_rate: float = 0.5
    dtype: Any = jnp.float32

    # Depends on model-type
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_layers: int = 12
    num_heads: int = 12
    head_size: int = 64

    # Run-dependent
    max_seq_length: int = 256

    @classmethod
    def from_model_name(cls, model_name, max_seq_length):
        if model_name == "roberta-base":
            return cls(
                hidden_size=768,
                intermediate_size=3072,
                num_layers=12,
                num_heads=12,
                head_size=64,
                max_seq_length=max_seq_length,
            )
        elif model_name == "roberta-large":
            return cls(
                hidden_size=1024,
                intermediate_size=4096,
                num_layers=24,
                num_heads=16,
                head_size=64,
                max_seq_length=max_seq_length,
            )
        else:
            raise KeyError(model_name)


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


def learning_rate_scheduler(lr, total_steps):
    def f(step):
        return lr * (total_steps - step) / total_steps
    return f


def save_params(params, path):
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(params))


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


def prepare_dataset(model_name, task, max_seq_length):
    task_dataset = datasets.load_dataset("glue", name=task)
    tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

    def tokenize_examples(examples: dict):
        return tokenizer(
            *[examples[field] for field in INPUT_FIELDS_DICT[task]],
            padding="max_length",
            max_length=max_seq_length,
            truncation="longest_first",
        )
    tokenized_dataset = task_dataset.map(tokenize_examples, batched=True)
    if task == "mnli":
        return {
            "train": tokenized_dataset["train"],
            "validation": tokenized_dataset["validation_matched"],
        }
    else:
        return {
            "train": tokenized_dataset["train"],
            "validation": tokenized_dataset["validation"],
        }


def convert_batch(raw_batch, reshape_for_devices=True):
    batch_size = len(raw_batch["input_ids"][0])
    local_device_count = jax.local_device_count()
    full_batch =  {
        "input_ids": np.array(raw_batch["input_ids"]),
        "attention_mask": np.array(raw_batch["attention_mask"]),
        "label": np.array(raw_batch["label"]),
    }
    if reshape_for_devices:
        batch = {
            k: v.reshape((local_device_count, -1) + v.shape[1:])
            for k, v in full_batch.items()
        }
        return batch
    else:
        return full_batch


def compute_batch_metrics(logits, labels, task):
    if task in ["qnli", "mnli", "rte", "sst", "wnli"]:
        metrics = score_task(logits, labels, task)
    else:
        # The other metrics aren't jax-friendly
        metrics = {}
    metrics["loss"] = cross_entropy_loss(logits, labels)
    return metrics


def score_task(logits, labels, task):
    if task in ["qnli", "mnli", "rte", "sst", "wnli"]:
        preds = logits.argmax(-1)
        return {
            "accuracy": (preds == labels).mean(),
        }
    elif task == "stsb":
        preds = logits[:, -1]
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }
    elif task in ["mrpc", "qqp"]:
        preds = logits.argmax(-1)
        acc = (preds == labels).mean()
        labels = np.array(labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }
    elif task == "cola":
        preds = logits.argmax(-1)
        return {
            "mcc": matthews_corrcoef(labels, preds),
        }
    elif task == "sst":
        return
    else:
        raise KeyError(task)


def train_step(state, batch, task):
    def loss_fn(params):
        logits_ = state.apply_fn(
            {"params": params},
            batch["input_ids"],
            batch["attention_mask"],
            False,
        )
        if task == "stsb":
            loss = ((logits_[:, 0] - batch["label"]) ** 2).mean()
        else:
            loss = cross_entropy_loss(logits_, batch["label"])
        return loss, logits_

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, 'batch')
    state = state.apply_gradients(grads=grads)
    metrics = compute_batch_metrics(logits, batch['label'], task=task)
    return state, metrics


def eval_step(state, batch):
    logits = state.apply_fn(
        {"params": state.params},
        batch["input_ids"],
        batch["attention_mask"],
        True,
    )
    return logits


multi_device_train_step = jax.pmap(train_step, axis_name="batch", static_broadcasted_argnums=(2,))
multi_device_eval_step = jax.pmap(eval_step, axis_name="batch")
single_device_eval_step = jax.jit(eval_step)


def train_epoch(state, train_ds, task, batch_size):
    batch_metrics = []
    num_examples = len(train_ds)
    # Skip the last batch. Too annoying to deal with :/
    num_used_examples = num_examples // batch_size * batch_size
    permuted_idx = np.random.permutation(num_examples)[:num_used_examples]
    for i in tqdm.trange(0, num_used_examples, batch_size):
        batch_idx = permuted_idx[i:i + batch_size]
        batch = convert_batch(train_ds[batch_idx])
        state, metrics = multi_device_train_step(state, batch, task)
        batch_metrics.append(metrics)
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    return state, epoch_metrics_np


def eval_model(state, validation_ds, task, eval_batch_size):
    num_examples = len(validation_ds)
    all_logits = []
    for i in tqdm.trange(0, num_examples, eval_batch_size):
        raw_batch = validation_ds[i:i + eval_batch_size]
        if len(raw_batch["input_ids"]) == eval_batch_size:
            # Regular batch
            batch = convert_batch(raw_batch)
            logits = jax.device_get(multi_device_eval_step(state, batch))
            logits = logits.reshape(-1, logits.shape[-1])
        else:
            # Special handling for last batch.
            # I guess we'll just loop over it?
            # If we really wanted to be efficient, we could do some of this last
            # batch across all devices first.
            local_device_count = jax.local_device_count()
            capacity_per_device = eval_batch_size // local_device_count
            sub_logits_ls = []
            for j in range(0, len(raw_batch["input_ids"]), capacity_per_device):
                single_device_state = jax_utils.unreplicate(state)
                raw_sub_batch = validation_ds[i + j: i + j + capacity_per_device]
                sub_batch = convert_batch(
                    raw_sub_batch,
                    reshape_for_devices=False,
                )
                sub_logits = jax.device_get(
                    single_device_eval_step(single_device_state, sub_batch)
                )
                sub_logits_ls.append(sub_logits)
            logits = np.concatenate(sub_logits_ls, axis=0)

        all_logits.append(logits)
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.array(validation_ds["label"])
    return score_task(all_logits, all_labels, task)


def main(_):
    cfg = FLAGS
    os.makedirs(cfg.output_dir, exist_ok=True)
    model_config = RoBERTaConfig.from_model_name(cfg.model_name, max_seq_length=cfg.max_seq_length)

    # Prepare the dataset
    tokenized_dataset = prepare_dataset(
        model_name=cfg.model_name,
        task=cfg.task,
        max_seq_length=cfg.max_seq_length,
    )

    # Create our model
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    task_model = RoBERTaClassificationModel(
        config=model_config,
        num_labels=NUM_LABELS_DICT[cfg.task],
    )

    # Create dummy inputs to help initialize parameters
    dummy_input_ids = jnp.ones([1, cfg.max_seq_length]).astype(jnp.int32)
    dummy_mask = jnp.ones([1, cfg.max_seq_length])
    params = task_model.init(init_rng, dummy_input_ids, dummy_mask)['params']

    # Insert pretrained RoBERTa encoder params
    urllib.request.urlretrieve(
        f"https://huggingface.co/{cfg.model_name}/resolve/main/pytorch_model.bin",
        os.path.join(cfg.output_dir, f"{cfg.model_name}.p"),
    )
    roberta_params = load_params_from_pt_weights(
        os.path.join(cfg.output_dir, f"{cfg.model_name}.p"),
        config=model_config,
    )
    params = insert_roberta_params(params, roberta_params)

    # Set up our optimizer and training state
    total_steps = cfg.num_epochs * math.ceil(len(tokenized_dataset["train"]) / cfg.batch_size)
    tx = optax.adamw(learning_rate_scheduler(cfg.learning_rate, total_steps=total_steps))
    state = train_state.TrainState.create(apply_fn=task_model.apply, params=params, tx=tx)

    # Start training
    epoch_metrics = []
    for epoch_i in tqdm.trange(cfg.num_epochs):
        state, metrics = train_epoch(
            state=state,
            train_ds=tokenized_dataset["train"],
            task=cfg.task,
            batch_size=cfg.batch_size,
        )
        epoch_metrics.append(metrics)
        eval_metrics = eval_model(
            state=state,
            validation_ds=tokenized_dataset["validation"],
            task=cfg.task,
            eval_batch_size=cfg.batch_size * 2,
        )
        print(f"Epoch {epoch_i}", eval_metrics)
        save_params(params, os.path.join(cfg.outdir_dir, f"model__epoch{epoch_i}.params"))


if __name__ == '__main__':
    flags.mark_flags_as_required(['task', 'output_dir'])
    app.run(main)
