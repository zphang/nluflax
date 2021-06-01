import jax.numpy as jnp


def onehot(labels, num_classes):
    x = (labels[..., None] == jnp.arange(num_classes)[None])
    return x.astype(jnp.float32)


def cross_entropy_loss(logits, labels):
    return -jnp.mean(jnp.sum(onehot(labels, num_classes=logits.shape[-1]) * logits, axis=-1))
