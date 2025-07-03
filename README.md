# CMB Delensing with Deep Learning and Rotation Augmentation

A deep learning framework for Cosmic Microwave Background (CMB) delensing using U-Net architecture with rotation-based data augmentation. This project implements state-of-the-art techniques to reconstruct unlensed CMB maps from gravitationally lensed observations.

## 🌌 Project Overview

Gravitational lensing distorts the primordial CMB signal, making it challenging to extract cosmological information. This project uses deep learning to "delens" CMB maps, recovering the original unlensed signal. The approach employs:

- **U-Net Architecture**: A convolutional neural network optimized for image-to-image translation tasks
- **Rotation Augmentation**: Spherical rotation transformations to increase training data diversity
- **Multi-GPU Training**: Distributed training across multiple GPUs for faster convergence
- **HEALPix Integration**: Native support for HEALPix spherical pixelization scheme

## ✨ Key Features

- **🔄 Rotation Augmentation**: Advanced spherical rotation techniques for robust training
- **🚀 Multi-GPU Support**: Distributed training on up to 8 GPUs
- **💾 Checkpoint Management**: Automatic saving and resuming of training sessions
- **📊 Comprehensive Logging**: Detailed training metrics and loss tracking
- **🎯 Flexible Architecture**: Configurable U-Net depth and parameters
- **🌐 HEALPix Compatible**: Full integration with HEALPix spherical data format

## 📋 Requirements

### System Requirements
- Linux/Unix operating system
- NVIDIA GPU(s) with CUDA support
- Python 3.7+
- At least 48GB RAM (recommended for large datasets)

### Python Dependencies
```
numpy>=1.19.0
tensorflow>=2.8.0
scipy>=1.7.0
healpy>=1.15.0  # For HEALPix operations
matplotlib>=3.3.0  # For visualization
```

### Hardware Recommendations
- **GPU**:  Tesla A40
- **RAM**: 64GB+ for optimal performance
- **Storage**: SSD with >500GB free space for datasets and checkpoints

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd CMB_Delensing
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU Setup**
   ```python
   import tensorflow as tf
   print("GPUs Available:", tf.config.list_physical_devices('GPU'))
   ```

## 📁 Project Structure

```
CMB_Delensing/
├── run_rot.py                 # Main training script
├── unet/                      # U-Net model implementation
│   └── unet_2d.py            # 2D U-Net architecture
├── train_dataset/            # Training data directory
│   ├── Blen_5maps.npy       # Lensed B-mode maps
│   ├── Bunl_5maps.npy       # Unlensed B-mode maps
│   ├── Elen_5maps.npy       # Lensed E-mode maps
│   └── Eunl_5maps.npy       # Unlensed E-mode maps
├── rots_npys/               # Rotation transformation arrays
│   ├── rot_arr_*.npy        # Forward rotation arrays
│   └── rerot_arr_*.npy      # Inverse rotation arrays
├── checkpoints/             # Model checkpoints
│   ├── check_B/            # B-mode checkpoints
│   └── check_E/            # E-mode checkpoints
├── result/                  # Training outputs
└── README.md               # This file
```

## 🚀 Usage

### Basic Training Command

```bash
python run_rot.py <epochs> <field> <scale>
```

**Parameters:**
- `epochs`: Number of training epochs (e.g., 2000)
- `field`: CMB field type ("B" for B-mode, "E" for E-mode)
- `scale`: Scaling factor for output normalization (typically 1-10)

### Example Commands

```bash
# Train B-mode delensing for 2000 epochs with scale factor 5
python run_rot.py 2000 B 5

# Train E-mode delensing for 1500 epochs with scale factor 3  
python run_rot.py 1500 E 3

# Short test run
python run_rot.py 100 B 1
```

### Training Configuration

The script automatically configures:
- **Learning Rate Schedule**: Piecewise decay (1e-2 → 1e-3 → 1e-4)
- **Batch Size**: 16 (adjustable in code)
- **GPU Usage**: All available GPUs (0-7)
- **Rotation Parameters**: Predefined theta/phi combinations

### Resuming Training

Training automatically resumes from the last checkpoint if interrupted:
```bash
# Same command - will auto-resume from last checkpoint
python run_rot.py 2000 B 5
```

## ⚙️ Configuration Options

### Model Architecture Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_filters` | 256 | Number of filters in first layer |
| `network_depth` | 3 | U-Net encoder/decoder depth |
| `dropout` | 0.2 | Dropout rate for regularization |
| `growth_factor` | 2 | Filter multiplication factor per layer |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Training batch size |
| `learning_rate` | Scheduled | Piecewise constant decay |
| `loss` | 'logcosh' | Training loss function |
| `optimizer` | Adam | Optimization algorithm |

### Data Augmentation

The rotation augmentation uses 8 predefined combinations:
- **Theta**: [60°, 120°, 240°]  
- **Phi**: [-60°, -30°, 30°, 60°]

Currently uses combination index 2: `[120°, 30°]`

## 📊 Output Files

### Training Results
- `{field}_pred_epochs_{epochs}_{theta}_{phi}.npy`: Model predictions
- `{field}_test_epochs_{epochs}_{theta}_{phi}.npy`: Test labels  
- `{field}_label_epochs_{epochs}_{theta}_{phi}.npy`: Test inputs
- `{field}_loss_train_epochs_{epochs}_{theta}_{phi}.npy`: Training losses
- `{field}_loss_val_epochs_{epochs}_{theta}_{phi}.npy`: Validation losses

### Model Checkpoints
- `checkpoint_100.ckpt`: Model weights (saved every 100 epochs)
- `final_model.h5`: Complete trained model
- `training_status.npy`: Training state for resumption

## 📈 Monitoring Training

### Real-time Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor training logs
tail -f training.log
```

### Performance Metrics
- **Training Loss**: Log-cosh loss on training set
- **Validation Loss**: Log-cosh loss on validation set  
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory Error**
```bash
# Reduce batch size in code
batch_size = 8  # Instead of 16
```

**2. CUDA Out of Memory**
```bash
# Limit GPU memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

**3. Import Errors**
```bash
# Ensure unet module is in Python path
export PYTHONPATH="${PYTHONPATH}:."
```

**4. Data Loading Issues**
- Verify dataset files exist in `train_dataset/`
- Check HEALPix rearrangement files in `/home/nisl/Data/CMB_DeLensing/`
- Ensure rotation arrays exist in `rots_npys/`

### Performance Optimization

**Multi-GPU Training**
```python
# Verify distributed strategy
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")
```

**Memory Optimization**
```python
# Enable mixed precision training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

## 📚 Technical Details

### Algorithm Overview

1. **Data Loading**: Load lensed/unlensed CMB map pairs
2. **Rotation Augmentation**: Apply spherical rotations using HEALPix
3. **Model Training**: Train U-Net to map lensed → unlensed
4. **Inverse Rotation**: Transform predictions back to original coordinates
5. **Evaluation**: Compare predictions with ground truth

### U-Net Architecture

```
Input (512×512×1)
    ↓
Encoder: 3 levels of conv+pooling
    ↓
Bottleneck: Dense feature representation  
    ↓
Decoder: 3 levels of upconv+concat
    ↓
Output (512×512×1)
```

### Data Flow

```
Raw HEALPix → Rotation → 512×512 Maps → U-Net → Predictions → Inverse Rotation → Results
```



## 📖 References

- **HEALPix**: Hierarchical Equal Area isoLatitude Pixelization
- **U-Net**: Convolutional Networks for Biomedical Image Segmentation
- **CMB Lensing**: Gravitational lensing of the cosmic microwave background



## 👤 Author

**nisl**
- Cosmic Microwave Background research
- Deep learning applications in cosmology

