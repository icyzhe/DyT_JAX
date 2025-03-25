import equinox as eqx
import jax.numpy as jnp
import chex

class DynamicTanh(eqx.Module):
    alpha: float = 0.5
    weight: chex.Array
    bias: chex.Array
    channels_last: bool
    normalized_shape: chex.Shape

    def __init__(self, normalized_shape, channels_last):
        self.normalized_shape = normalized_shape
        self.alpha = jnp.array(self.alpha)
        self.weight = jnp.ones(normalized_shape)
        self.bias = jnp.zeros(normalized_shape)
        self.channels_last = channels_last

    
    def __call__(self, x):
        x = jnp.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            weight = self.weight[:, None, None]
            bias = self.bias[:, None, None]
            x = x * weight + bias
        return x
    
    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha}, channels_last={self.channels_last}"


