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
- At least 32GB RAM (recommended for large datasets)

### Python Dependencies
```
numpy>=1.19.0
tensorflow>=2.8.0
scipy>=1.7.0
healpy>=1.15.0  # For HEALPix operations
matplotlib>=3.3.0  # For visualization
```

### Hardware Recommendations
- **GPU**: NVIDIA RTX 3080/4080 or Tesla V100/A100
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
   print("Available GPUs:", tf.config.list_physical_devices('GPU'))
   ```

## 📁 Project Structure

```
CMB_Delensing/
├── run_rot.py                 # Main training script
├── model/                     # Model implementation directory
│   ├── unet2d.py             # 2D U-Net architecture
│   └── unet2p.py             # U-Net++ architecture
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
```python
# Limit GPU memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

**3. Data Loading Errors**
```bash
# Check data file paths
ls -la train_dataset/
ls -la rots_npys/
```

**4. Checkpoint Issues**
```bash
# Clear corrupted checkpoints
rm -rf checkpoints/check_*
```

## 📚 Technical Details

### U-Net Architecture Features

1. **Encoder Path**: 
   - Progressive downsampling to extract high-level features
   - Uses stride-2 convolutions for downsampling
   - Filter count increases by growth_factor

2. **Decoder Path**:
   - Progressive upsampling to recover spatial resolution
   - Uses transpose convolutions for upsampling
   - Skip connections preserve spatial details

3. **Regularization Techniques**:
   - Batch normalization accelerates training
   - Dropout prevents overfitting
   - L2 regularization stabilizes training

### Rotation Data Augmentation

1. **HEALPix Processing**:
   - Convert spherical data to planar grids
   - Apply rotation transformations for data diversity
   - Use inverse transformations to recover original coordinates

2. **Rotation Strategy**:
   - Pre-computed rotation matrices for efficiency
   - Multi-angle combinations cover different orientations
   - Maintain physical meaning consistency

## 🎯 Best Practices

### Training Recommendations

1. **Data Preprocessing**:
   - Ensure input data is properly normalized
   - Check data quality and completeness
   - Set appropriate scaling factors

2. **Hyperparameter Tuning**:
   - Start with small-scale training for testing
   - Monitor training/validation loss trends
   - Adjust learning rate based on convergence

3. **Resource Management**:
   - Monitor GPU memory usage
   - Regularly clean unnecessary checkpoints
   - Ensure sufficient storage space

### Performance Optimization

1. **Batch Size Adjustment**:
   - Adjust batch size based on GPU memory
   - Larger batches usually provide more stable training
   - Consider gradient accumulation techniques

2. **Learning Rate Strategy**:
   - Use learning rate schedulers
   - Monitor loss plateaus
   - Consider adaptive learning rates

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact

- Author: nisl
- Project Link: [https://github.com/username/CMB_Delensing](https://github.com/username/CMB_Delensing)

## 🙏 Acknowledgments

Thanks to the following projects and resources:
- TensorFlow team for the deep learning framework
- HEALPix project for spherical data processing tools
- CMB research community for theoretical guidance and data support 