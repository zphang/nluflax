from flax import linen as nn
import jax.numpy as jnp
from typing import Any


import nluflax.models.encoders as encoders
import nluflax.models.heads as heads
from nluflax.models.common.losses import cross_entropy_loss


def make_classification_model_fn(encoder_model, classification_head):
    def f(input_kwargs):
        encoder_output = encoder_model(**input_kwargs)
        head_kwargs = {
            "pooled": encoder_output["output"],
        }
        if "deterministic" in input_kwargs:
            head_kwargs = input_kwargs["deterministic"]
        logits = classification_head(**head_kwargs)
        return logits
    return f


def make_classification_loss_fn(encoder_model, classification_head):
    def f(input_kwargs, labels):
        logits = make_classification_model_fn(
            encoder_model=encoder_model,
            classification_head=classification_head,
        )(input_kwargs=input_kwargs)
        loss = cross_entropy_loss(
            logits=logits,
            labels=labels
        )
        return loss
    return f
