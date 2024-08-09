import pytest
import jax
import numpy as np
from jax import numpy as jnp
import jax._src.test_util as jtu

from jax.test_util import check_grads
from causal_conv1d_jax import causal_conv1d
from utils import causal_conv1d_ref

import torch
from einops import rearrange

jax.config.update("jax_enable_x64", True)

def l2_loss(x, weight, b, silu_activation=False, channel_last=False):
    predictions = causal_conv1d(x, weight, b, activation = silu_activation, channel_last = channel_last)
    return -jnp.mean(predictions**2)

# ================================
#          SANITY CHECK
# ================================
@pytest.mark.parametrize("dtype", [np.float32, np.float16, jnp.bfloat16])
@pytest.mark.parametrize("dtype2", [None, np.float32, np.float16, jnp.bfloat16])
def test_causal_conv1d(dtype, dtype2):
    dtype2 = dtype if dtype2 is None else dtype2
    x = np.ones((1,2,6), dtype=dtype)
    w = np.ones((2,3), dtype=dtype2)
    b = np.zeros((2), dtype=dtype2)

    out = jax.jit(causal_conv1d)(x, w, b)
    print(out, out.shape)

@pytest.mark.parametrize("dtype", [np.float32, np.float16, jnp.bfloat16])
@pytest.mark.parametrize("dtype2", [None, np.float32, np.float16, jnp.bfloat16])
def test_causal_conv1d_bwd(dtype, dtype2):
    dtype2 = dtype if dtype2 is None else dtype2

    x = np.ones((1,2,6), dtype=dtype)
    w = np.ones((2,3), dtype=dtype2)
    b = np.zeros((2), dtype=dtype2)

    #check_grads(loss, (x, w, b), modes=["rev"], order=1) # Error for float16
    
    g_x = jax.grad(l2_loss, argnums=0)(x, w, b)
    print("Gradient", g_x, g_x.shape)

# ===============================
#          UNIT TEST
# ===============================

@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("itype", [np.float32, np.float16, jnp.bfloat16])
@pytest.mark.parametrize("silu_activation", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize(
    "seqlen", [8, 16, 32, 64, 128, 151, 256, 372, 512, 784, 1024, 1134, 2048, 4096]
)
@pytest.mark.parametrize('dim', [64, 4096 + 32])
def unit_test_causal_conv1d(TEST_NAME, dim, seqlen, width, has_bias, silu_activation, itype, channel_last):
    # set seed
    key = jax.random.PRNGKey(0)
    rtol, atol = (3e-4, 1e-3) if itype == np.float32 else (3e-3, 5e-3)
    if itype == jnp.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)    

    # === INIT INPUT VARIABLES ===
    batch = 2
    key, subkey = jax.random.split(key)
    if not channel_last:
        x = jax.random.normal(subkey, shape=(batch, 4096 + dim + 64, seqlen), dtype=itype)[:, 4096:4096 + dim, :]
    else:
        print(">>Channellast")
        x = rearrange(
            jax.random.normal(subkey, shape=(batch, seqlen, 4096 + dim + 64), dtype=itype)[:, :, 4096:4096 + dim], "b s d -> b d s"
        )
    key, subkey = jax.random.split(key)
    weight = jax.random.normal(subkey, shape = (dim, width), dtype=np.float32)
    key, subkey = jax.random.split(key)
    if has_bias:
        bias = jax.random.normal(subkey, shape = (dim, ), dtype=np.float32)
    else:
        bias = None
    x_ref = torch.tensor(np.array(x), requires_grad=True).cuda()
    weight_ref = torch.tensor(np.array(weight), requires_grad=True).cuda()
    bias_ref = torch.tensor(np.array(bias), requires_grad=True).cuda()
    activation = None if not silu_activation else "silu"

    # === TEST FWD CONSISTENCY WITH REFERENCE ===
    out = jax.jit(causal_conv1d)(x, weight, bias, activation=silu_activation, channel_last=channel_last)
    out_ref = causal_conv1d_ref(x_ref, weight_ref, bias_ref, activation=activation)

    print(f"{TEST_NAME} Output max diff: {jnp.abs(out - out_ref.detach().cpu().numpy()).max()}")
    print(f"{TEST_NAME} Output mean diff: {jnp.abs(out - out_ref.detach().cpu().numpy()).mean()}")

    # === TEST BWD CONSISTENCY WITH REFERENCE ===
    g_x, g_weight = jax.grad(l2_loss, argnums=(0,1))(x, weight, bias, silu_activation, channel_last)

    mse = (-(out_ref-0)**2).mean()
    #mse = torch.nn.MSELoss()(out_ref, torch.zeros_like(out_ref))
    x_ref.retain_grad()
    weight_ref.retain_grad()
    bias_ref.retain_grad()
    mse.backward()

    #print("g_w (1)", g_weight[:5,:5])
    #print(weight_ref.grad[:5,:5])

    print(f"{TEST_NAME} dx max diff: {jnp.abs(g_x - x_ref.grad.detach().cpu().numpy()).max()}")
    print(f"{TEST_NAME} dweight max diff: {jnp.abs(g_weight - weight_ref.grad.detach().cpu().numpy()).max()}")
    if has_bias:
        g_bias = jax.grad(l2_loss, argnums=2)(x, weight, bias, silu_activation, channel_last)
        print(f"{TEST_NAME} dbias max diff: {jnp.abs(g_bias - bias_ref.grad.detach().cpu().numpy()).max()}")

    assert jnp.allclose(g_x, x_ref.grad.cpu().numpy().astype(np.float32), rtol=rtol, atol=atol)
    assert jnp.allclose(g_weight, weight_ref.grad.cpu().numpy(), rtol=rtolw, atol=atolw)
    if has_bias:
        assert jnp.allclose(g_bias, bias_ref.grad.cpu().numpy(), rtol=rtolw, atol=atolw)

if __name__ == "__main__":
    # === Test activation ===
    for silu in [False, True]:
        unit_test_causal_conv1d(f"[CAUSAL_CONV_1D TEST SILU {silu}]", 64, 8, 2, True,
                                silu_activation=silu,
                                itype=np.float32, channel_last=False)
        print()

    for channellast in [True]:
        unit_test_causal_conv1d(f"[CAUSAL_CONV_1D TEST CHANNELLAST {channellast}]", 64, 8, 2, True,
                                silu_activation=False,
                                itype=np.float32, channel_last=channellast)
        print()
    
    for dtype in [np.float16]: # Test jnp.bfloat16 with torch complicated, TODO: compare with np.float16
        unit_test_causal_conv1d(f"[CAUSAL_CONV_1D TEST DTYOE {dtype}]", 64, 8, 2, True,
                                silu_activation=False,
                                itype=dtype, channel_last=False)
        print()

    """
    # === Sanity checks ===
    test_causal_conv1d(np.float32)
    test_causal_conv1d_bwd(np.float32)

    test_causal_conv1d(np.float16)
    test_causal_conv1d_bwd(np.float16)

    test_causal_conv1d(jnp.bfloat16)
    test_causal_conv1d_bwd(jnp.bfloat16)

    # Mixed
    test_causal_conv1d(np.float32, jnp.bfloat16)
    test_causal_conv1d_bwd(np.float32, jnp.bfloat16)
    """