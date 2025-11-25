# Monitoring Nemotron Parse Training

This guide covers how to monitor evaluations and metrics during Nemotron Parse finetuning.

## Quick Start

### 1. Console Logging (Built-in)

The training script logs metrics to console by default. You'll see output like:

```
Epoch 1/3
Step 10/100: loss=2.345, learning_rate=2e-5
Step 20/100: loss=2.123, learning_rate=2e-5
...
Eval: eval_loss=1.987, eval_char_accuracy=0.85, eval_word_accuracy=0.78
```

**Configure logging frequency:**
```yaml
# training/configs/nemotron_config.yaml
nemotron:
  logging_steps: 10      # Log every 10 steps
  eval_steps: 100        # Evaluate every 100 steps
```

### 2. TensorBoard (Recommended)

TensorBoard is enabled by default and provides real-time visualization.

**Start training:**
```bash
python -m src.training.train_nemotron \
    --train-data data/train.json \
    --output-dir trained_models/nemotron
```

**View metrics:**
```bash
# In another terminal
tensorboard --logdir trained_models/nemotron
# Open http://localhost:6006 in your browser
```

**What you'll see:**
- **Scalars**: Loss, learning rate, character/word accuracy over time
- **Training vs Validation**: Compare train/eval metrics
- **GPU Utilization**: Monitor GPU memory and compute usage

### 3. MLflow (Production Tracking)

MLflow provides experiment tracking, model versioning, and comparison.

**Setup:**

1. Start MLflow server (optional, for UI):
```bash
mlflow ui --port 5000
# Or use Azure ML workspace tracking URI
```

2. Configure in config file:
```yaml
# config/default_config.yaml
logging:
  mlflow:
    enabled: true
    tracking_uri: "http://localhost:5000"  # or Azure ML workspace URI
    experiment_name: "nemotron-finetuning"
```

3. Or set environment variable:
```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="nemotron-finetuning"
```

4. Run training:
```bash
python -m src.training.train_nemotron \
    --train-data data/train.json \
    --config training/configs/nemotron_config.yaml
```

**View in MLflow UI:**
- Open `http://localhost:5000`
- Compare runs side-by-side
- View hyperparameters, metrics, and artifacts
- Download best model checkpoints

### 4. Weights & Biases (Optional)

For advanced experiment tracking with wandb:

**Install:**
```bash
pip install wandb
wandb login
```

**Enable in training:**
```python
# Modify train_nemotron.py or set environment variable
export WANDB_PROJECT="nemotron-finetuning"
export WANDB_WATCH="all"  # Track gradients and parameters
```

**Update training args:**
```python
# In create_training_arguments function
report_to=["wandb", "tensorboard"]
```

## Metrics Explained

### Training Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| `loss` | Cross-entropy loss | Decreasing, typically 0.5-2.0 |
| `learning_rate` | Current learning rate | Follows schedule |
| `epoch` | Current epoch | 1 to num_epochs |
| `step` | Training step | Incremental |

### Evaluation Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| `eval_loss` | Validation loss | Lower is better, should track train loss |
| `eval_char_accuracy` | Character-level accuracy | > 0.90 for good OCR |
| `eval_word_accuracy` | Word-level accuracy | > 0.85 for good OCR |
| `eval_exact_match_rate` | Exact text match | > 0.70 for structured docs |

### Model Metrics

| Metric | Description | Notes |
|--------|-------------|-------|
| `trainable_params` | Number of trainable parameters | LoRA: ~1-5% of total |
| `total_params` | Total model parameters | ~900M for Nemotron Parse |
| `trainable_ratio` | % of parameters being trained | LoRA: 0.01-0.05 |

## Monitoring During Training

### Real-time Console Output

```bash
# Run training and watch console
python -m src.training.train_nemotron --train-data train.json

# Output example:
# ============================================================
# Nemotron Parse Finetuning
# ============================================================
# 
# Loading processor from nvidia/NVIDIA-Nemotron-Parse-v1.1...
# Loading model...
# 
# Model Statistics:
#   Total parameters: 900,000,000
#   Trainable parameters: 4,500,000
#   Trainable ratio: 0.50%
# 
# Starting training...
# Epoch 1/3
#   10%|██        | 10/100 [00:45<06:45, loss=2.345]
#   20%|████      | 20/100 [01:30<05:30, loss=2.123]
# ...
# Eval: eval_loss=1.987, eval_char_accuracy=0.8523
```

### TensorBoard Dashboard

**Key tabs to monitor:**

1. **SCALARS** - Main metrics
   - `train/loss` - Should decrease smoothly
   - `eval/loss` - Should track train loss (watch for overfitting)
   - `eval/eval_char_accuracy` - Should increase over time
   - `eval/eval_word_accuracy` - Should increase over time

2. **GRAPHS** - Model architecture (if enabled)

3. **HISTOGRAMS** - Parameter distributions (if enabled)

**Red flags to watch for:**
- Eval loss increasing while train loss decreases → Overfitting
- Loss not decreasing → Learning rate too high/low
- GPU memory errors → Reduce batch size or use QLoRA

### MLflow Experiment Tracking

**Compare multiple runs:**

```python
import mlflow

# View all runs
runs = mlflow.search_runs(experiment_names=["nemotron-finetuning"])
print(runs[["run_id", "metrics.eval_char_accuracy", "params.learning_rate"]])

# Get best run
best_run = runs.loc[runs["metrics.eval_char_accuracy"].idxmax()]
print(f"Best accuracy: {best_run['metrics.eval_char_accuracy']}")
```

**View in UI:**
- Compare hyperparameters across runs
- Plot metrics over time
- Download model artifacts from best runs

## Advanced Monitoring

### Custom Metrics Callback

Add custom metrics during training:

```python
from transformers import TrainerCallback

class CustomMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        # Calculate custom metrics
        # Log to MLflow, wandb, etc.
        pass

# Add to trainer callbacks
trainer.add_callback(CustomMetricsCallback())
```

### GPU Monitoring

Monitor GPU usage during training:

```bash
# In another terminal
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i 1
```

**What to watch:**
- GPU memory usage (should be stable)
- GPU utilization (should be 80-100% during training)
- Temperature (should stay < 80°C)

### Log File Monitoring

Training logs are saved to the output directory:

```bash
# Watch logs in real-time
tail -f trained_models/nemotron/training.log

# Or view checkpoint logs
cat trained_models/nemotron/checkpoint-500/trainer_state.json
```

## Troubleshooting

### Metrics Not Appearing

**Issue:** No eval metrics in TensorBoard/MLflow

**Solutions:**
1. Check `eval_steps` is set correctly in config
2. Ensure validation dataset is provided
3. Verify `compute_metrics` function is working
4. Check logs for errors during evaluation

### Loss Not Decreasing

**Possible causes:**
- Learning rate too high → Try 1e-5
- Learning rate too low → Try 5e-5
- Data format incorrect → Check dataset loading
- Model not training → Verify trainable parameters > 0

### Overfitting Detection

**Signs:**
- Train loss decreasing, eval loss increasing
- Eval accuracy plateauing or decreasing

**Solutions:**
- Increase `lora_dropout` (try 0.1)
- Reduce `num_epochs`
- Increase `early_stopping_patience`
- Add more training data

## Example Monitoring Workflow

```bash
# Terminal 1: Start training
python -m src.training.train_nemotron \
    --train-data data/train.json \
    --val-data data/val.json \
    --output-dir trained_models/nemotron \
    --config training/configs/nemotron_config.yaml

# Terminal 2: Start TensorBoard
tensorboard --logdir trained_models/nemotron --port 6006

# Terminal 3: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 4: Monitor logs
tail -f trained_models/nemotron/training.log
```

**In browser:**
- TensorBoard: http://localhost:6006
- MLflow UI: http://localhost:5000 (if enabled)

## Best Practices

1. **Always enable TensorBoard** - It's lightweight and provides immediate feedback
2. **Use MLflow for production** - Better for experiment comparison and model versioning
3. **Monitor GPU utilization** - Ensure you're using GPU efficiently
4. **Set appropriate eval_steps** - Balance between monitoring frequency and training speed
5. **Save checkpoints frequently** - Use `save_steps` to save progress
6. **Compare multiple runs** - Use MLflow to find best hyperparameters

## Configuration Examples

### Minimal Monitoring (Console Only)
```yaml
nemotron:
  logging_steps: 50
  eval_steps: 500
```

### Standard Monitoring (TensorBoard + Console)
```yaml
nemotron:
  logging_steps: 10
  eval_steps: 100
```

### Full Monitoring (MLflow + TensorBoard + Console)
```yaml
nemotron:
  logging_steps: 10
  eval_steps: 100

logging:
  mlflow:
    enabled: true
    tracking_uri: "http://localhost:5000"
    experiment_name: "nemotron-finetuning"
```

