# Dynamic Layer Range Implementation - Quick Reference

## Summary of Changes

This implements **dynamic, configurable layer targeting** for the LLaVA attention scaling mechanism. Instead of hardcoding layer attention scaling to all 32 layers, you can now select specific layer ranges to test the "Middle Layer Hypothesis" (whether spatial reasoning occurs in layers 11-22).

---

## Files Modified

### 1. **model_zoo/llama/modeling_llama_add_attn.py** (Core Changes)

- **Line ~655**: Added `self.target_layers = range(0, config.num_hidden_layers)` to `LLaMAModel.__init__`
- **Line ~677**: Added `target_layers: Optional[range] = None` parameter to `LLaMAModel.forward()`
- **Line ~710**: Added logic to use provided `target_layers` or fall back to model's default
- **Line ~756**: Pass `target_layers=target_layers` to decoder layer loop
- **Line ~365**: Added `target_layers: Optional[range] = None` parameter to `LLaMADecoderLayer.forward()`
- **Line ~385**: Pass `target_layers=target_layers` to `self.self_attn()`
- **Line ~217**: Added `target_layers: Optional[range] = None` parameter to `LLaMAAttention.forward()`
- **Line ~268-270**: **CRITICAL CHANGE**: Replaced hardcoded `if idx<32:` with dynamic `if idx in target_layers:`
  - Old: `if idx<32:` (all layers applied)
  - New: `if idx in target_layers:` (only specified layers applied)

### 2. **model_zoo/llava/modeling_llava_scal.py**

- **Line ~293**: Added `target_layers: Optional[range] = None` parameter to `LlavaForConditionalGenerationScal.forward()`
- **Line ~463**: Pass `target_layers=target_layers` to `self.language_model()` call

### 3. **model_zoo/llava15.py**

- **Line ~57**: Added `target_layers: Optional[range] = None` parameter to `_add_weight_greedy_search()`
- **Line ~147**: Pass `target_layers=target_layers` to model forward call
- **Line ~316**: Added `target_layers=None` parameter to `get_out_scores_wh_batched()`
- **Lines 397-405**: Pass `target_layers=target_layers` to all `self.model.generate()` calls

### 4. **main_aro.py** (CLI Interface)

- **Lines 14-19**: Added `parse_layer_range()` function to convert "0-10" format to `range(0, 11)`
- **Line 40**: Added `--target-layers` CLI argument with help text
- **Line 44**: Parse target_layers after argument parsing
- **Lines 68, 75, 79**: Pass `args.target_layers` to all `get_out_scores_wh_batched()` calls

---

## Usage

### Basic Commands

**Run with all layers (baseline, default behavior):**

```bash
python main_aro.py \
  --dataset Controlled_Images_A \
  --method adapt_vis \
  --weight1 0.5 \
  --weight2 1.5 \
  --threshold 0.4 \
  --option four \
  --seed 1
```

**Run with early layers only (0-10):**

```bash
python main_aro.py \
  --dataset Controlled_Images_A \
  --method adapt_vis \
  --weight1 0.5 \
  --weight2 1.5 \
  --threshold 0.4 \
  --option four \
  --seed 1 \
  --target-layers 0-10
```

**Run with middle layers only (11-22):**

```bash
python main_aro.py \
  --dataset Controlled_Images_A \
  --method adapt_vis \
  --weight1 0.5 \
  --weight2 1.5 \
  --threshold 0.4 \
  --option four \
  --seed 1 \
  --target-layers 11-22
```

**Run with late layers only (23-31):**

```bash
python main_aro.py \
  --target-layers 23-31 \
  [other args...]
```

---

## Format for --target-layers

The `--target-layers` argument accepts a simple range format:

- **`0-10`** → scales layers 0, 1, 2, ..., 10 (11 layers)
- **`11-22`** → scales layers 11, 12, ..., 22 (12 layers)
- **`23-31`** → scales layers 23, 24, ..., 31 (9 layers)
- **`0-31`** → scales all 32 layers (baseline)
- **`None`** or omitted → uses model's default (all layers)

---

## Testing the Layer Ablation Hypothesis

Use the provided test script to automatically run all 4 layer configurations:

```bash
python test_layer_ablation.py
```

This runs:

1. Early layers (0-10): Baseline perception
2. Middle layers (11-22): Core reasoning hypothesis
3. Late layers (23-31): Token decoding
4. All layers (0-31): Current baseline

### Expected Outcomes

| Scenario                      | Interpretation                                               |
| ----------------------------- | ------------------------------------------------------------ |
| `middle_layers ≈ all_layers`  | ✅ Spatial reasoning is mid-stage (confirms hypothesis)      |
| `all_layers >> middle_layers` | ❌ Early layers are critical (refutes hypothesis)            |
| `early_layers ≈ all_layers`   | ❌ Layers 0-10 sufficient for task (refute depth hypothesis) |

---

## Technical Details

### Layer Targeting Logic

```python
# Old (hardcoded):
if idx<32:  # applies to ALL layers
    attn_weights[:, :, mask] *= weight

# New (dynamic):
if idx in target_layers:  # applies ONLY to specified layers
    attn_weights[:, :, mask] *= weight
```

### Call Chain

```
main_aro.py
  ↓ (parse --target-layers)
  ↓ parse_layer_range("11-22") → range(11, 23)
  ↓
LlavaWrapper.get_out_scores_wh_batched(..., target_layers=range(11, 23))
  ↓
self.model.generate(..., target_layers=range(11, 23))
  ↓
_add_weight_greedy_search(..., target_layers=range(11, 23))
  ↓
self() [LlavaForConditionalGenerationScal.forward(..., target_layers=range(11, 23))]
  ↓
self.language_model(..., target_layers=range(11, 23))
  ↓
LLaMAModel.forward(..., target_layers=range(11, 23))
  ↓
for idx, decoder_layer in enumerate(self.layers):
    layer_outputs = decoder_layer(..., target_layers=target_layers, idx=idx)
  ↓
LLaMADecoderLayer.forward(..., target_layers=target_layers, idx=idx)
  ↓
self.self_attn(..., target_layers=target_layers, idx=idx)
  ↓
LLaMAAttention.forward(..., target_layers=target_layers, idx=idx)
  ↓
if idx in target_layers:  ← Dynamic decision point
    attn_weights[:, :, mask] *= weight
```

---

## Backward Compatibility

✅ **Fully backward compatible**

- If `--target-layers` is omitted, uses model's default: `range(0, 32)` (all layers)
- Existing commands without `--target-layers` work unchanged
- Default behavior matches previous hardcoded `if idx<32:`

---

## Output

Results are saved to `outputs/res.json` in JSONL format with fields:

- `dataset`: e.g., "Controlled_Images_A"
- `model`: e.g., "llava1.5"
- `option`: "four", "six", etc.
- `method`: e.g., "adapt_vis"
- `weight`: global weight (typically 1.0 for adapt_vis)
- `"Individual accuracy"`: accuracy score (0.0-1.0)
- `correct_id`: indices of correct predictions

**Note**: `outputs/res.json` stores `weight` (global scaling) but not `weight1`, `weight2`, `threshold`, or `target_layers`. You'll need to reconstruct these from submission logs or add metadata to the save_scores function if needed.

---

## Integration Notes

The implementation threads `target_layers` through the entire call stack from CLI argument to the attention mechanism's innermost scaling operation. Each component:

1. Accepts it as an optional parameter
2. Defaults to `None` if not provided
3. Passes it downstream unchanged
4. Only the `LLaMAAttention.forward()` uses it to make the scaling decision

This design ensures minimal disruption to existing code while enabling precise control over which layers are scaled.
