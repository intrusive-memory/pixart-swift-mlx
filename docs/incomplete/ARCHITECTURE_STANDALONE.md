# pixart-swift-mlx — Implementation Architecture

**Status**: DRAFT — derived from reference implementation research.
**Companion to**: `REQUIREMENTS.md` (product-level spec, P1–P16).
**Purpose**: Internal architecture detail sufficient to write code without constantly consulting the PyTorch reference. All tensor shapes use MLX NHWC convention unless noted otherwise.

---

## A1. PixArt DiT Transformer — Internal Architecture

### A1.1 Model Configuration

| Parameter | Value |
|---|---|
| `hidden_size` | 1152 |
| `num_heads` | 16 |
| `attention_head_dim` | 72 (1152 / 16) |
| `depth` | 28 blocks |
| `patch_size` | 2 |
| `in_channels` | 4 (VAE latent channels) |
| `out_channels` | 8 (4 noise + 4 variance, `pred_sigma=True`) |
| `mlp_ratio` | 4.0 (FFN hidden = 4608) |
| `caption_channels` | 4096 (T5-XXL embedding dim) |
| `model_max_length` | 120 tokens |

### A1.2 Patch Embedding

Operation: `Conv2d(4, 1152, kernel_size=2, stride=2)` followed by flatten and transpose.

```
Input:  [B, H/8, W/8, 4]       — e.g. [B, 128, 128, 4] for 1024px
Conv2d: [B, H/16, W/16, 1152]  — e.g. [B, 64, 64, 1152]
Flatten: [B, T, 1152]          — e.g. [B, 4096, 1152] where T = (H/16)*(W/16)
```

**Position embeddings are 2D sinusoidal, NOT learned.** They are recomputed dynamically per forward pass based on actual spatial dimensions. This enables variable resolution natively.

- Grid coordinates: `arange(grid_size) / (grid_size / base_size) / pe_interpolation`
- Half the dimensions encode height, half encode width
- Frequencies: `1 / 10000^(2i/d)`, standard sinusoidal
- `base_size` = `input_size // patch_size` (e.g., 16 for reference 512px training)
- `pe_interpolation` = 2 for PixArt-Sigma XL

### A1.3 DiT Block Structure

Each of the 28 blocks contains, in order: **Self-Attention → Cross-Attention → FFN**.

Components per block:
- `norm1`: `LayerNorm(1152, elementwise_affine=False)` — no learnable gamma/beta
- `attn`: Self-attention (Q/K/V from image tokens)
- `cross_attn`: Cross-attention (Q from image, K/V from text)
- `norm2`: `LayerNorm(1152, elementwise_affine=False)` — no learnable gamma/beta
- `mlp`: `Linear(1152, 4608) → GELU(tanh) → Linear(4608, 1152)`
- `scale_shift_table`: Learned parameter of shape `(6, 1152)` — the AdaLN-Zero parameters

### A1.4 AdaLN-Zero Conditioning (AdaLN-Single Design)

**This is the key architectural innovation vs vanilla DiT.** Instead of each block having its own MLP to produce conditioning parameters, there is ONE global MLP (`t_block`) that produces a single conditioning vector. Each block then adds its own learned `scale_shift_table` to this shared vector.

The 6 parameters per block:
1. `shift_msa` — additive shift before self-attention
2. `scale_msa` — multiplicative scale before self-attention
3. `gate_msa` — multiplicative gate after self-attention (zero-initialized)
4. `shift_mlp` — additive shift before FFN
5. `scale_mlp` — multiplicative scale before FFN
6. `gate_mlp` — multiplicative gate after FFN (zero-initialized)

Modulation function: `t2i_modulate(x, shift, scale) = x * (1 + scale) + shift`

**Cross-attention receives NO AdaLN modulation.** No shift/scale/gate from the timestep.

### A1.5 DiT Block Forward Pass (Exact)

```
forward(x, y, t, mask):
    B, N, C = x.shape  // e.g. (B, 4096, 1152)

    // Unpack 6 modulation params: timestep + per-block learned table
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp =
        (scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

    // 1. Self-Attention with AdaLN-Zero
    x = x + gate_msa * attn(t2i_modulate(norm1(x), shift_msa, scale_msa))

    // 2. Cross-Attention (NO modulation, NO norm on query)
    x = x + cross_attn(x, y, mask)

    // 3. FFN with AdaLN-Zero
    x = x + gate_mlp * mlp(t2i_modulate(norm2(x), shift_mlp, scale_mlp))

    return x
```

### A1.6 Self-Attention

Standard multi-head attention with optional QK normalization:

```
qkv = Linear(1152, 3*1152)(x)  // or separate Q, K, V projections in diffusers format
q, k, v = chunk(qkv, 3)
q = reshape(q, [B, T, 16, 72]).transpose(0, 2, 1, 3)  // [B, 16, T, 72]
k = reshape(k, [B, T, 16, 72]).transpose(0, 2, 1, 3)
v = reshape(v, [B, T, 16, 72]).transpose(0, 2, 1, 3)

// Optional QK norm (PixArt-Sigma enables this)
q = LayerNorm(q)
k = LayerNorm(k)

output = MLXFast.scaledDotProductAttention(q, k, v, scale: 1/sqrt(72))
output = output.transpose(0, 2, 1, 3).reshape(B, T, 1152)
output = Linear(1152, 1152)(output)  // output projection
```

### A1.7 Cross-Attention

Q from image tokens, K/V from pre-projected text embeddings:

```
q = Linear(1152, 1152)(x)          // query from image
kv = Linear(1152, 2*1152)(y)       // key+value from text (already 1152-dim)
k, v = chunk(kv, 2)

// Same multi-head reshape as self-attention
output = scaledDotProductAttention(q, k, v, scale: 1/sqrt(72), mask: text_mask)
output = Linear(1152, 1152)(output)  // output projection, zero-initialized at init
```

**Text projection happens BEFORE the blocks**, not inside cross-attention:
```
CaptionProjection: Linear(4096, 1152) → GELU(tanh) → Linear(1152, 1152)
```
This converts T5 embeddings from `[B, seq_len, 4096]` → `[B, seq_len, 1152]`.

### A1.8 Timestep Conditioning Pipeline

**Stage 1 — Sinusoidal embedding:**
```
Input: scalar timestep t, shape [B]
freqs = exp(-log(10000) * arange(128) / 128)
embedding = [cos(t * freqs), sin(t * freqs)]  // [B, 256]
```

**Stage 2 — MLP projection:**
```
Linear(256, 1152) → SiLU → Linear(1152, 1152)  // [B, 1152]
```

**Stage 3 — Micro-conditions (PixArt-Sigma specific):**
```
csize_embedder: height, width → sinusoidal(256) each → MLP(256→384→384) each → concat → [B, 768]
ar_embedder: aspect_ratio → sinusoidal(256) → MLP(256→384→384) → [B, 384]
t = t + concat([csize, ar])  // [B, 1152]
```
Note: `384 = 1152 / 3`. The resolution embedder produces 2 × 384 = 768, the AR embedder produces 384, total = 1152.

**Stage 4 — Block conditioning projection:**
```
t_block: SiLU → Linear(1152, 6*1152)  // [B, 6912]
```
This output is passed to every block and combined with each block's `scale_shift_table`.

**Stage 5 — Final layer conditioning:** The raw timestep embedding `t` (shape `[B, 1152]`, before `t_block`) is passed separately to the final layer.

### A1.9 Final Layer / Unpatchify

```
norm_final = LayerNorm(1152, elementwise_affine=False)
linear = Linear(1152, patch_size² × out_channels)  // Linear(1152, 2×2×8 = 32)
scale_shift_table = Parameter(shape: [2, 1152])     // only 2 params (shift+scale, no gate)

forward(x, t):
    shift, scale = (scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
    x = t2i_modulate(norm_final(x), shift, scale)
    x = linear(x)  // [B, T, 32]
    return x
```

**Unpatchify:**
```
x: [B, T, 32] → reshape [B, H, W, 2, 2, 8]
→ einsum rearrange to [B, 8, H*2, W*2]  // or NHWC equivalent
→ [B, H*2, W*2, 8]
```

When `pred_sigma=True`, the 8-channel output is split: first 4 channels = noise prediction, last 4 = variance prediction (discarded at inference).

### A1.10 Key Differences from Vanilla DiT

| Aspect | Vanilla DiT | PixArt-Sigma |
|---|---|---|
| Conditioning source | Per-block MLP from class+timestep | Global `t_block` MLP + per-block learned `scale_shift_table` |
| Text conditioning | Class label embedding | Cross-attention to T5 tokens |
| Cross-attn norm | N/A | No norm applied to query |
| Position embedding | Fixed sin-cos (square grids) | Dynamic sin-cos with `pe_interpolation` for arbitrary aspect ratios |
| Micro-conditions | None | Resolution + aspect ratio embeddings added to timestep |
| Architecture | Plain sequential stack | Plain sequential stack (no U-Net skip connections) |

---

## A2. T5-XXL Encoder — Internal Architecture

### A2.1 Model Dimensions

| Parameter | Value |
|---|---|
| `d_model` (hidden size) | 4096 |
| `d_ff` (FFN intermediate) | 10240 |
| `d_kv` (head dimension) | 64 |
| `num_heads` | 64 |
| `inner_dim` (num_heads × d_kv) | 4096 |
| `num_layers` | 24 |
| `vocab_size` | 32128 |
| `relative_attention_num_buckets` | 32 |
| `relative_attention_max_distance` | 128 |
| `layer_norm_epsilon` | 1e-6 |
| `feed_forward_proj` | `gated-gelu` (GeGLU) |
| `pad_token_id` | 0 |
| `eos_token_id` | 1 |

### A2.2 Normalization: RMSNorm (NOT standard LayerNorm)

T5 uses a custom norm with no mean subtraction and no bias:

```
variance = mean(x², dim=-1, keepdim=True)  // computed in float32
output = weight * (x / sqrt(variance + eps))
```

- Weight shape: `[4096]`
- Variance computation MUST be in float32 for numerical stability
- Use `MLXNN.RMSNorm` or `MLXFast.rmsNorm`

### A2.3 Encoder Block Structure (Pre-Norm)

Each of the 24 blocks has 2 sub-layers with pre-norm residual connections:

```
Input x
  │
  ├─→ RMSNorm(x) → SelfAttention → + x  (residual)
  │                                  │
  ├─→ RMSNorm(x) → GeGLU FFN    → + x  (residual)
  │                                  │
  Output
```

After all 24 blocks: `final_layer_norm` (RMSNorm) → output.

Full encoder flow:
```
token_ids → embedding lookup → [block_0 ... block_23] → final_RMSNorm → output [B, S, 4096]
```

### A2.4 Attention — CRITICAL: No Scaling

**T5 does NOT divide attention scores by sqrt(d_k).** The relative position bias was trained with unscaled scores. This means you CANNOT use `MLXFast.scaledDotProductAttention` with the standard scale — you must pass `scale: 1.0` or implement attention manually.

```
scores = Q @ K^T                    // [B, 64, S, S] — NO division by sqrt(64)
scores += position_bias             // [1, 64, S, S] — added before softmax
scores += attention_mask            // additive mask (0 for attend, -inf for block)
attn = softmax(scores, dtype=float32)  // softmax in float32, then cast back
output = attn @ V
output = reshape → Linear(4096, 4096)  // output projection
```

All Q/K/V/O projections: `Linear(4096, 4096, bias: false)` — **no bias in T5 attention**.

Tensor shapes through attention:
```
Input:   [B, S, 4096]
Q/K/V:   [B, S, 4096] → reshape [B, S, 64, 64] → transpose [B, 64, S, 64]
Scores:  [B, 64, S, S]
Output:  [B, 64, S, 64] → transpose/reshape [B, S, 4096]
O proj:  [B, S, 4096]
```

### A2.5 Relative Position Bias

T5 uses **learned relative position bias** instead of any positional embeddings. There are NO sinusoidal or learned absolute position embeddings.

**Only layer 0** has the bias embedding: `Embedding(32, 64)` — 32 buckets, 64 heads. The computed bias is reused by all subsequent layers.

**Bucketing algorithm (bidirectional for encoder):**
```
relative_position = memory_position - context_position  // [S, S] matrix

if bidirectional:
    num_buckets_per_direction = 16  // 32 / 2
    bucket = (relative_position > 0) * 16  // direction offset
    abs_position = abs(relative_position)

max_exact = 8  // 16 / 2 — exact buckets for positions 0-7
is_small = abs_position < max_exact

// Logarithmic buckets for larger distances (8-128+)
large_bucket = max_exact + (
    log(abs_position / max_exact) / log(max_distance / max_exact) * (16 - max_exact)
)
large_bucket = min(large_bucket, 15)  // clamp to num_buckets_per_direction - 1

bucket += where(is_small, abs_position, large_bucket)
```

**Bias computation:**
```
context_pos = arange(query_len)[:, None]    // [Q, 1]
memory_pos = arange(key_len)[None, :]       // [1, K]
relative_pos = memory_pos - context_pos     // [Q, K]
buckets = relative_position_bucket(relative_pos)  // [Q, K]
values = embedding_lookup(buckets)          // [Q, K, 64]
bias = values.permute(2, 0, 1).unsqueeze(0) // [1, 64, Q, K]
```

The bias is added to raw attention scores before softmax.

### A2.6 FFN: GeGLU (Gated Linear Unit with GELU)

T5 v1.1 uses gated activation with three weight matrices (not two):

```
gate = gelu_new(Linear(4096, 10240, bias=false)(x))   // wi_0
value = Linear(4096, 10240, bias=false)(x)              // wi_1
hidden = gate * value                                    // element-wise
output = Linear(10240, 4096, bias=false)(hidden)         // wo
```

**Activation**: `gelu_new` is the tanh approximation of GELU:
```
f(x) = 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

For exact parity with HuggingFace, use tanh-approximated GELU. For practical purposes, the difference vs exact GELU is negligible at inference.

### A2.7 Tokenizer

- **Type**: SentencePiece Unigram (`spiece.model`)
- **Vocab size**: 32128
- **Special tokens**: `<pad>` = 0, `</s>` (EOS) = 1, `<unk>` = 2
- **No BOS token** — T5 does not use beginning-of-sequence
- **EOS appended** automatically
- **Padding**: Pad to `max_length=120` with token 0
- **Output**: `input_ids [B, 120]` and `attention_mask [B, 120]` (1 = real, 0 = padding)

The attention mask is passed both to T5 and separately to the DiT as `encoder_attention_mask`.

### A2.8 Implementation Gotchas

1. **No attention scaling.** Pass `scale: 1.0` — the position bias was trained with unscaled scores.
2. **Position bias shared across layers.** Compute once in layer 0, reuse for layers 1-23.
3. **Softmax in float32.** Compute softmax in float32, cast back to working dtype.
4. **RMSNorm variance in float32.** Always compute variance in float32.
5. **No bias in any Linear layer.** All Q/K/V/O and FFN projections have `bias=false`.
6. **Embedding NOT tied.** `shared.weight` is not tied to any output head.
7. **Attention mask conversion.** Tokenizer mask (1=attend, 0=pad) must be converted to additive format (0.0=attend, -inf=block) before adding to position bias.

---

## A3. SDXL VAE Decoder — Internal Architecture

### A3.1 Configuration

| Parameter | Value |
|---|---|
| `latent_channels` | 4 |
| `output_channels` | 3 (RGB) |
| `block_out_channels` | [128, 256, 512, 512] |
| `layers_per_block` | 2 (decoder uses 3 = layers_per_block + 1) |
| `norm_num_groups` | 32 |
| `scaling_factor` | 0.13025 |
| `sample_size` | 1024 |

### A3.2 Complete Decoder Pipeline

For a 1024x1024 image with latents `[B, 128, 128, 4]` (NHWC):

```
 1. z = z / 0.13025                                          // un-scale
 2. z = post_quant_proj(z)          -> [B, 128, 128, 4]       // Linear(4, 4) — 1x1 conv as Linear
 3. x = conv_in(z)                  -> [B, 128, 128, 512]     // Conv2d(4, 512, 3x3, pad=1)
 4. x = mid_block_resnet_0(x)       -> [B, 128, 128, 512]     // ResNet 512->512
 5. x = mid_block_attention(x)      -> [B, 128, 128, 512]     // Single-head self-attention
 6. x = mid_block_resnet_1(x)       -> [B, 128, 128, 512]     // ResNet 512->512
 7. x = up_block_0(x)               -> [B, 256, 256, 512]     // 3xResNet 512->512 + upsample 2x
 8. x = up_block_1(x)               -> [B, 512, 512, 512]     // 3xResNet 512->512 + upsample 2x
 9. x = up_block_2(x)               -> [B, 1024, 1024, 256]   // 3xResNet 512->256 + upsample 2x
10. x = up_block_3(x)               -> [B, 1024, 1024, 128]   // 3xResNet 256->128, NO upsample
11. x = GroupNorm(32, 128)(x)        -> [B, 1024, 1024, 128]
12. x = SiLU(x)                     -> [B, 1024, 1024, 128]
13. x = conv_out(x)                  -> [B, 1024, 1024, 3]    // Conv2d(128, 3, 3x3, pad=1)
```

### A3.3 ResNet Block

```
forward(x):
    y = GroupNorm(32, in_ch)(x)
    y = SiLU(y)
    y = Conv2d(in_ch, out_ch, 3x3, pad=1)(y)
    y = GroupNorm(32, out_ch)(y)
    y = SiLU(y)
    y = Conv2d(out_ch, out_ch, 3x3, pad=1)(y)

    if in_ch != out_ch:
        x = Linear(in_ch, out_ch)(x)  // 1x1 conv as Linear, weight squeezed from [out,in,1,1]

    return x + y  // residual
```

- **GroupNorm** with 32 groups everywhere, `pytorch_compatible=True`
- **SiLU** activation (not GELU, not ReLU)
- **No time embedding** in VAE ResNets (unlike UNet ResNets)
- **No dropout**
- Skip connection uses Linear only when channels change

### A3.4 Mid-Block Attention

Single-head self-attention at the bottleneck (512 channels):

```
forward(x):
    B, H, W, C = x.shape           // NHWC
    y = GroupNorm(32, 512)(x)
    q = Linear(512, 512)(y).reshape(B, H*W, C)
    k = Linear(512, 512)(y).reshape(B, H*W, C)
    v = Linear(512, 512)(y).reshape(B, H*W, C)

    scale = 1 / sqrt(512)
    scores = (q * scale) @ k.T      // [B, HW, HW]
    attn = softmax(scores)
    y = (attn @ v).reshape(B, H, W, C)
    y = Linear(512, 512)(y)
    return x + y                     // residual
```

- **Single-head** (heads = 512 / 512 = 1)
- Only appears in mid-block — no attention in up-blocks
- Attention projections have bias (unlike T5)

### A3.5 Upsampling

Nearest-neighbor 2x interpolation followed by convolution (NOT transposed convolution):

```
// Nearest-neighbor 2x via broadcast+reshape
x: [B, H, W, C]
x = broadcast_to(x[:, :, None, :, None, :], [B, H, 2, W, 2, C])
x = reshape(x, [B, H*2, W*2, C])

// Then convolution
x = Conv2d(C, C, 3x3, pad=1)(x)
```

Alternatively, use `MLXNN.Upsample(scaleFactor: 2, mode: .nearest)` if available.

Up-blocks 0, 1, 2 have upsampling. Up-block 3 (final) does not.

### A3.6 Up-Block Channel Progression

```
reversed_channels = [512, 512, 256, 128]
input_channels    = [512, 512, 512, 256]  // prepended with channels[0]

Up-block 0: in=512, out=512  — 3 ResNets (all 512->512), upsample
Up-block 1: in=512, out=512  — 3 ResNets (all 512->512), upsample
Up-block 2: in=512, out=256  — ResNet 512->256 (with shortcut), 2x ResNet 256->256, upsample
Up-block 3: in=256, out=128  — ResNet 256->128 (with shortcut), 2x ResNet 128->128, NO upsample
```

### A3.7 VAE is NOT Quantized

The VAE stays in float16. Conv2d layers do not benefit from weight-only quantization. Total VAE decoder size: ~160 MB in float16.

---

## A4. DPM-Solver++ Scheduler — Internal Architecture

### A4.1 PixArt-Sigma Scheduler Configuration

```
algorithm_type:        "dpmsolver++"
beta_start:            0.0001
beta_end:              0.02
beta_schedule:         "linear"         // NOT cosine, NOT shifted
num_train_timesteps:   1000
prediction_type:       "epsilon"        // model predicts noise
solver_order:          2
solver_type:           "midpoint"       // (vs "heun")
lower_order_final:     true
timestep_spacing:      "linspace"
thresholding:          false
num_inference_steps:   20               // pipeline default
guidance_scale:        4.5              // pipeline default
```

**IMPORTANT**: Despite the original requirements mentioning "shifted cosine schedule," PixArt-Sigma's released checkpoint uses a **standard linear beta schedule** with no shift. The SNR-shift code exists in the training codebase but is disabled. The inference schedule is a plain linear schedule.

### A4.2 Noise Schedule Computation

```
betas = linspace(0.0001, 0.02, 1000)            // [1000]
alphas = 1.0 - betas                              // [1000]
alphas_cumprod = cumprod(alphas)                   // [1000]

// Derived quantities
alpha_t = sqrt(alphas_cumprod)                     // signal coefficient
sigma_t = sqrt(1 - alphas_cumprod)                 // noise coefficient
lambda_t = log(alpha_t) - log(sigma_t)             // half-logSNR
sigmas = sqrt((1 - alphas_cumprod) / alphas_cumprod)  // diffusers sigma convention
```

### A4.3 Timestep Selection

With `timestep_spacing="linspace"` and 20 steps:
```
timesteps = linspace(0, 999, 21).round()[::-1][:-1]  // 20 values, descending
// ~ [999, 949, 899, 849, ..., 49]

sigmas = interpolate(timesteps, arange(0, 1000), full_sigmas)
sigmas = concatenate([sigmas, [0.0]])  // append 0 for final step
```

### A4.4 Conversion: Epsilon to x0 Prediction

DPM-Solver++ works in data-prediction space. The model output (epsilon) is converted to x0:

```
// Diffusers sigma convention conversion
alpha_t = 1 / sqrt(sigma^2 + 1)
sigma_t = sigma * alpha_t

x0_pred = (sample - sigma_t * model_output) / alpha_t
```

### A4.5 First-Order Update (Steps 0 and 19)

```
sigma_t, sigma_s = sigmas[step+1], sigmas[step]
alpha_t, sigma_t_val = sigma_to_alpha_sigma(sigma_t)
alpha_s, sigma_s_val = sigma_to_alpha_sigma(sigma_s)
lambda_t = log(alpha_t) - log(sigma_t_val)
lambda_s = log(alpha_s) - log(sigma_s_val)
h = lambda_t - lambda_s

x_t = (sigma_t_val / sigma_s_val) * sample - alpha_t * (exp(-h) - 1.0) * model_output
```

Use `expm1(-h)` instead of `exp(-h) - 1` for numerical stability.

### A4.6 Second-Order Multistep Update (Steps 1-18, Midpoint)

```
sigma_t  = sigmas[step+1]    // target
sigma_s0 = sigmas[step]       // current
sigma_s1 = sigmas[step-1]     // previous

lambda_t, lambda_s0, lambda_s1 = log(alpha/sigma) for each

m0 = model_outputs[-1]   // most recent x0 prediction
m1 = model_outputs[-2]   // previous x0 prediction

h   = lambda_t - lambda_s0
h_0 = lambda_s0 - lambda_s1
r0  = h_0 / h

D0 = m0                          // 0th order difference
D1 = (1/r0) * (m0 - m1)          // 1st order finite difference

// Midpoint method:
x_t = (sigma_t / sigma_s0) * sample
    - alpha_t * expm1(-h) * D0
    - 0.5 * alpha_t * expm1(-h) * D1
```

### A4.7 Classifier-Free Guidance

CFG is applied **before** the scheduler step, in the pipeline loop:

```
// 1. Duplicate latents
latent_input = concat([latents, latents])  // [2B, C, H, W]

// 2. Run model on both conditional and unconditional
noise_pred = transformer(latent_input, ...)

// 3. Split and apply CFG
noise_pred_uncond, noise_pred_text = chunk(noise_pred, 2)
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

// 4. Discard learned sigma (take only first 4 of 8 channels)
noise_pred = noise_pred[:, :4, :, :]

// 5. Pass to scheduler
latents = scheduler.step(noise_pred, timestep, latents)
```

PixArt-Sigma uses constant `guidance_scale=4.5` (no dynamic scaling).

### A4.8 Scheduler State

```swift
// Mutable state between steps:
var modelOutputs: [MLXArray?]     // ring buffer of size solver_order (2)
var lowerOrderNums: Int           // counts up to solver_order during warmup
var stepIndex: Int                // current position in timestep schedule

// Precomputed (immutable after setTimesteps):
let timesteps: [Int]
let sigmas: [Float]               // length = num_inference_steps + 1

// Precomputed (immutable after init):
let betas: [Float]
let alphas: [Float]
let alphasCumprod: [Float]
```

Reset between generations by calling `setTimesteps()`.

### A4.9 Step Order for 20-Step Generation

- Step 0: 1st-order (bootstrapping — no previous model outputs)
- Steps 1-18: 2nd-order multistep (midpoint)
- Step 19 (final): 1st-order (`lower_order_final=true`)

---

## A5. MLX Swift Idioms and Patterns

### A5.1 Module Definition Pattern

```swift
public class MyLayer: Module, @unchecked Sendable {
    @ModuleInfo var linear: Linear        // @ModuleInfo enables quantization
    @ModuleInfo var norm: RMSNorm
    let config: MyConfig                   // non-module properties

    public init(config: MyConfig) {
        self._linear.wrappedValue = Linear(config.inputDim, config.outputDim, bias: false)
        self._norm.wrappedValue = RMSNorm(dimensions: config.inputDim, eps: 1e-6)
        self.config = config
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm(linear(x))
    }
}
```

- Use `@ModuleInfo` on any `Linear` that should be quantizable at runtime
- Use `let` for arrays of sub-modules (reflection discovers them)
- Conform to `@unchecked Sendable` for concurrent access
- Forward pass is `callAsFunction` (makes model callable directly)

### A5.2 Weight Loading Pattern

```swift
let weights = try loadArrays(url: fileURL)                    // [String: MLXArray]
let mapped = weights.flatMap { mapper($0.key, $0.value.asType(.float16)) }
let params = ModuleParameters.unflattened(mapped)
try model.update(parameters: params, verify: .noUnusedKeys)
```

Key mapping uses `String.replacingOccurrences(of:with:)` for renames. Fused QKV weights are split via array slicing: `value[0..<dim, 0...]`.

### A5.3 Quantization

```swift
// Quantize all Linear layers in a model to int4, group_size 64
quantize(model: transformer, groupSize: 64, bits: 4)
eval(model.parameters())  // materialize quantized weights

// QuantizedLinear stores: weight (packed uint32), scales, biases (quantization offsets)
// Uses quantizedMM() — optimized Metal kernel
```

### A5.4 Attention Pattern

```swift
let q = q.reshaped([B, S, numHeads, headDim]).transposed(0, 2, 1, 3)
let k = k.reshaped([B, S, numHeads, headDim]).transposed(0, 2, 1, 3)
let v = v.reshaped([B, S, numHeads, headDim]).transposed(0, 2, 1, 3)

let output = MLXFast.scaledDotProductAttention(
    queries: q, keys: k, values: v,
    scale: Float(1.0 / sqrt(Float(headDim))),
    mask: mask
)

let result = output.transposed(0, 2, 1, 3).reshaped([B, S, numHeads * headDim])
```

**For T5**: pass `scale: 1.0` since T5 uses unscaled attention.

### A5.5 Lazy Evaluation and eval()

MLX operations are lazy — nothing computes until `eval()`:

```swift
eval(latents)              // force computation
Memory.clearCache()        // free recycled GPU buffers
```

**Critical in denoising loop**: call `eval(latents)` after each scheduler step to prevent unbounded computation graph growth. Periodically call `Memory.clearCache()`.

### A5.6 Memory Management

```swift
Memory.activeMemory        // bytes in use
Memory.cacheMemory         // bytes in recycling pool
Memory.peakMemory          // high-water mark
Memory.clearCache()        // free cached buffers
Memory.cacheLimit = bytes  // cap the recycling pool
```

For phase-based loading, unload model by releasing all references and calling `Memory.clearCache()`.

Progressive weight mapping to free memory during load:
```swift
for key in allKeys {
    guard let value = weights.removeValue(forKey: key) else { continue }
    mapped[newKey] = value.asType(.float16)
}
```

### A5.7 Available Layer Types

- **Linear**: `Linear`, `QuantizedLinear`
- **Conv**: `Conv1d`, `Conv2d`, `Conv3d` (NHWC layout)
- **Norm**: `LayerNorm`, `RMSNorm`, `GroupNorm`, `BatchNorm`, `InstanceNorm`
- **Embedding**: `Embedding`, `QuantizedEmbedding`
- **Activations**: `GELU`, `SiLU`, `ReLU`, `GLU`, `Mish`, `Tanh`, `Sigmoid`, `Softmax`, etc.
- **Positional**: `RoPE`, `SinusoidalPositionalEncoding`, `ALiBi`
- **Upsampling**: `Upsample` (modes: `.nearest`, `.linear`, `.cubic`)
- **Fast ops**: `MLXFast.scaledDotProductAttention`, `MLXFast.rmsNorm`, `MLXFast.layerNorm`, `MLXFast.RoPE`

### A5.8 Data Layout

MLX uses **NHWC** (channels-last) for all spatial operations. All Conv2d weights from PyTorch (OIHW) must be transposed: `[out, in, kH, kW]` -> `transpose(0, 2, 3, 1)` -> `[out, kH, kW, in]`.

---

## A6. Weight Key Mappings

### A6.1 Transformer: Original -> Diffusers Format

**Global keys:**

| Original (PixArt repo) | Diffusers Format |
|---|---|
| `x_embedder.proj.{weight,bias}` | `pos_embed.proj.{weight,bias}` |
| `y_embedder.y_proj.fc1.{weight,bias}` | `caption_projection.linear_1.{weight,bias}` |
| `y_embedder.y_proj.fc2.{weight,bias}` | `caption_projection.linear_2.{weight,bias}` |
| `t_embedder.mlp.0.{weight,bias}` | `adaln_single.emb.timestep_embedder.linear_1.{weight,bias}` |
| `t_embedder.mlp.2.{weight,bias}` | `adaln_single.emb.timestep_embedder.linear_2.{weight,bias}` |
| `csize_embedder.mlp.0.{weight,bias}` | `adaln_single.emb.resolution_embedder.linear_1.{weight,bias}` |
| `csize_embedder.mlp.2.{weight,bias}` | `adaln_single.emb.resolution_embedder.linear_2.{weight,bias}` |
| `ar_embedder.mlp.0.{weight,bias}` | `adaln_single.emb.aspect_ratio_embedder.linear_1.{weight,bias}` |
| `ar_embedder.mlp.2.{weight,bias}` | `adaln_single.emb.aspect_ratio_embedder.linear_2.{weight,bias}` |
| `t_block.1.{weight,bias}` | `adaln_single.linear.{weight,bias}` |
| `final_layer.linear.{weight,bias}` | `proj_out.{weight,bias}` |
| `final_layer.scale_shift_table` | `scale_shift_table` |
| `pos_embed` | *(discarded — recomputed)* |
| `y_embedder.y_embedding` | *(discarded)* |

**Per-block keys (depth d = 0..27):**

| Original | Diffusers | Notes |
|---|---|---|
| `blocks.{d}.scale_shift_table` | `transformer_blocks.{d}.scale_shift_table` | [6, 1152] |
| `blocks.{d}.attn.qkv.weight` | **Split into 3** along dim 0: | [3456, 1152] -> 3x[1152, 1152] |
| | `transformer_blocks.{d}.attn1.to_q.weight` | chunk 0 |
| | `transformer_blocks.{d}.attn1.to_k.weight` | chunk 1 |
| | `transformer_blocks.{d}.attn1.to_v.weight` | chunk 2 |
| `blocks.{d}.attn.qkv.bias` | **Split into 3**: `...attn1.to_{q,k,v}.bias` | [3456] -> 3x[1152] |
| `blocks.{d}.attn.proj.{w,b}` | `transformer_blocks.{d}.attn1.to_out.0.{w,b}` | |
| `blocks.{d}.attn.q_norm.{w,b}` | `transformer_blocks.{d}.attn1.q_norm.{w,b}` | QK norm |
| `blocks.{d}.attn.k_norm.{w,b}` | `transformer_blocks.{d}.attn1.k_norm.{w,b}` | QK norm |
| `blocks.{d}.cross_attn.q_linear.{w,b}` | `transformer_blocks.{d}.attn2.to_q.{w,b}` | |
| `blocks.{d}.cross_attn.kv_linear.weight` | **Split into 2** along dim 0: | [2304, 1152] -> 2x[1152, 1152] |
| | `transformer_blocks.{d}.attn2.to_k.weight` | chunk 0 |
| | `transformer_blocks.{d}.attn2.to_v.weight` | chunk 1 |
| `blocks.{d}.cross_attn.kv_linear.bias` | **Split into 2**: `...attn2.to_{k,v}.bias` | |
| `blocks.{d}.cross_attn.proj.{w,b}` | `transformer_blocks.{d}.attn2.to_out.0.{w,b}` | |
| `blocks.{d}.mlp.fc1.{w,b}` | `transformer_blocks.{d}.ff.net.0.proj.{w,b}` | GEGLU |
| `blocks.{d}.mlp.fc2.{w,b}` | `transformer_blocks.{d}.ff.net.2.{w,b}` | |

### A6.2 T5-XXL: HuggingFace -> MLX Key Mapping

| HuggingFace Key | MLX Key |
|---|---|
| `shared.weight` | `wte.weight` |
| `encoder.block.{N}` | `encoder.layers.{N}` |
| `encoder.block.{N}.layer.0.SelfAttention.q.weight` | `encoder.layers.{N}.attention.query_proj.weight` |
| `encoder.block.{N}.layer.0.SelfAttention.k.weight` | `encoder.layers.{N}.attention.key_proj.weight` |
| `encoder.block.{N}.layer.0.SelfAttention.v.weight` | `encoder.layers.{N}.attention.value_proj.weight` |
| `encoder.block.{N}.layer.0.SelfAttention.o.weight` | `encoder.layers.{N}.attention.out_proj.weight` |
| `encoder.block.{N}.layer.0.layer_norm.weight` | `encoder.layers.{N}.ln1.weight` |
| `encoder.block.{N}.layer.1.DenseReluDense.wi_0.weight` | `encoder.layers.{N}.dense.wi_0.weight` |
| `encoder.block.{N}.layer.1.DenseReluDense.wi_1.weight` | `encoder.layers.{N}.dense.wi_1.weight` |
| `encoder.block.{N}.layer.1.DenseReluDense.wo.weight` | `encoder.layers.{N}.dense.wo.weight` |
| `encoder.block.{N}.layer.1.layer_norm.weight` | `encoder.layers.{N}.ln2.weight` |
| `encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight` | `relative_attention_bias.embeddings.weight` |
| `encoder.final_layer_norm.weight` | `encoder.ln.weight` |

**Keys to discard**: All `decoder.*` and `lm_head.*` (not present in T5EncoderModel safetensors).

### A6.3 SDXL VAE: Diffusers -> MLX Key Mapping

| Diffusers Key | MLX Key |
|---|---|
| `post_quant_conv.{weight,bias}` | `post_quant_proj.{weight,bias}` (squeezed [4,4,1,1]->[4,4]) |
| `decoder.mid_block.resnets.0.*` | `decoder.mid_blocks.0.*` |
| `decoder.mid_block.attentions.0.*` | `decoder.mid_blocks.1.*` |
| `decoder.mid_block.resnets.1.*` | `decoder.mid_blocks.2.*` |
| `*.to_q.{weight,bias}` | `*.query_proj.{weight,bias}` |
| `*.to_k.{weight,bias}` | `*.key_proj.{weight,bias}` |
| `*.to_v.{weight,bias}` | `*.value_proj.{weight,bias}` |
| `*.to_out.0.{weight,bias}` | `*.out_proj.{weight,bias}` |
| `decoder.up_blocks.{i}.upsamplers.0.conv.*` | `decoder.up_blocks.{i}.upsample.*` |
| `*.conv_shortcut.weight` | squeezed from [out,in,1,1] to [out,in] |

All Conv2d weights: transpose `[O,I,kH,kW]` -> `[O,kH,kW,I]` for MLX NHWC layout.

### A6.4 Int4 Quantization Key Format

When a Linear layer is quantized to int4 with group_size=64, each weight key becomes three keys:

| Original | Quantized | Shape |
|---|---|---|
| `layer.weight` `[M, N]` | `layer.weight` `[M, N/8]` uint32 | Packed 4-bit values |
| | `layer.scales` `[M, N/64]` float16 | Per-group scale |
| | `layer.biases` `[M, N/64]` float16 | Per-group zero-point |
| `layer.bias` `[M]` | `layer.bias` `[M]` float16 | Unchanged |

Note: `layer.biases` (plural) = quantization zero-points. `layer.bias` (singular) = linear bias term. Both can coexist.

**Dequantization**: `x_reconstructed = x_quantized * scale + bias`

**What gets quantized:**
- **Transformer**: All Linear layers (attention Q/K/V/O, FFN, caption projection, timestep embedder). Keep `scale_shift_table`, LayerNorm, Conv2d patch embed as float16.
- **T5**: All attention Q/K/V/O and FFN wi_0/wi_1/wo. Keep `shared.weight` (Embedding), RMSNorm weights, `relative_attention_bias` (Embedding) as float16.
- **VAE**: Not quantized. Keep entirely in float16.

---

## A7. Errata: Corrections to REQUIREMENTS.md

Based on the reference implementation research, the following items in REQUIREMENTS.md need correction:

1. **P5.1.4**: "shifted cosine schedule" -> The released PixArt-Sigma checkpoint uses a **standard linear beta schedule**, not a shifted cosine schedule. The SNR-shift code exists in the training codebase but is disabled for the released weights. Update to: "linear beta schedule (beta_start=0.0001, beta_end=0.02)."

2. **P2.1.4**: "sequence lengths up to 512 tokens" -> PixArt-Sigma actually uses `max_length=120` tokens for the T5 encoder, not 512. While T5 supports up to 512, the pipeline caps at 120 to match the trained model's expectations.

3. **P3.1.1**: "Hidden dimension: 1152" is correct. "16 attention heads" is correct. But the **head dimension is 72** (1152/16), not the 64 implied by comparison to standard transformers.

4. **P3.1.5**: The model outputs **8 channels** (4 noise + 4 learned variance), not just the latent dimension. The pipeline discards the last 4 channels at inference.

5. **P3 FFN**: The FFN in DiT blocks uses **GELU(tanh)** activation (also called GEGLU in the diffusers codebase), not standard GELU. The `fc1` weight projects to `2 x 4608` and is split for the gating mechanism, or equivalently, `fc1` projects to 4608 with a separate gating path.
