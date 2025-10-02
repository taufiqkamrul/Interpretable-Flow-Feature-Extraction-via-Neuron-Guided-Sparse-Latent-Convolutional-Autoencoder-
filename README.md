# Interpretable-Flow-Feature-Extraction-via-Neuron-Guided-Sparse-Latent-Convolutional-Autoencoder-

# Flow Field Autoencoder with Sparsity Analysis

## Project Overview

This project implements a Convolutional Neural Network (CNN) based autoencoder for analyzing and compressing 2D flow field data from fluid dynamics simulations. The model learns a compressed latent representation of velocity fields and includes extensive analysis of neuron sparsity and activation patterns.

## What This Project Does

1. **Data Processing**: Loads 2D velocity field snapshots and subtracts the mean flow to obtain fluctuations
2. **Autoencoder Training**: Trains a CNN autoencoder to compress flow fields from (384×192×2) to a 72-dimensional latent space
3. **Sparsity Analysis**: Analyzes which neurons in the latent space are most important for reconstruction
4. **Visualization**: Creates multiple visualizations including:
   - Velocity magnitude fields
   - Neuron activation patterns across temporal segments
   - Reconstruction error vs sparsity plots
   - Comparison of reconstructions using different neuron subsets

## Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
```
numpy
pandas
matplotlib
tensorflow>=2.4.0
keras
scikit-learn
tqdm
pickle
```

## Installation

### Step 1: Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd flow-field-autoencoder

# Or download and extract the ZIP file
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Required Packages
```bash
pip install numpy pandas matplotlib tensorflow scikit-learn tqdm
```

Or use a requirements file:
```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
tensorflow>=2.4.0
scikit-learn>=0.24.0
tqdm>=4.50.0
```

### Step 4: GPU Setup (Optional but Recommended)
For GPU acceleration, install TensorFlow GPU version:
```bash
pip install tensorflow-gpu
```

Ensure you have:
- CUDA Toolkit (compatible with your TensorFlow version)
- cuDNN library

## Data Setup

### Required Data Files

1. **Flow Field Data**: `flow_field_data0.pickle`
   - Format: Pickle file containing numpy array
   - Shape: (n_snapshots, 384, 192, 2)
   - Contents: Velocity field snapshots (u, v components)

2. **Mean Velocity Field**: `mode0.csv`
   - Format: CSV file (no header)
   - Contents: Mean velocity components (U, V) flattened

### Data Directory Structure
```
project/
├── data/
│   ├── flow_field_data0.pickle
│   └── mode0.csv
├── main_training.py
├── sparsity_analysis.py
├── README.md
└── requirements.txt
```

### Update File Paths
Edit the file paths in the code to match your data location:
```python
# In the code, update these paths:
fnstr = "/path/to/your/data/flow_field_data0.pickle"
mode0 = pd.read_csv("/path/to/your/data/mode0.csv", header=None, sep='\s+')
```

## How to Run

### 1. Training the Autoencoder

Run the main training script:
```bash
python main_training.py
```

This will:
- Load the flow field data
- Preprocess by subtracting mean flow
- Train the CNN autoencoder for 30 epochs
- Save model checkpoints (if configured)

**Training Parameters:**
- Epochs: 30
- Batch size: 32
- Train/test split: 80/20
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)

### 2. Running Sparsity Analysis

After training, run the analysis script:
```bash
python sparsity_analysis.py
```

This generates:
- Temporal segment activation plots (4 segments + full dataset)
- Reconstruction error vs sparsity curves
- Single snapshot reconstructions with top-k neurons

## Project Structure

```
project/
├── main_training.py          # Main autoencoder training script
├── sparsity_analysis.py      # Neuron sparsity analysis
├── data/                      # Data directory
│   ├── flow_field_data0.pickle
│   └── mode0.csv
├── models/                    # Saved models (created during training)
├── results/                   # Output visualizations
├── README.md                  # This file
└── requirements.txt          # Python dependencies
```

## Model Architecture

### Encoder
```
Input (384, 192, 2)
  ↓ Conv2D(16) + MaxPool
  ↓ Conv2D(8) + MaxPool
  ↓ Conv2D(8) + MaxPool
  ↓ Conv2D(8) + MaxPool
  ↓ Conv2D(4) + MaxPool
  ↓ Conv2D(4) + MaxPool
  ↓ Flatten + Dense
Latent Space (72)
```

### Decoder
```
Latent Space (72)
  ↓ Dense + Reshape
  ↓ UpSampling + Conv2D(4)
  ↓ UpSampling + Conv2D(8)
  ↓ UpSampling + Conv2D(8)
  ↓ UpSampling + Conv2D(8)
  ↓ UpSampling + Conv2D(16)
  ↓ UpSampling + Conv2D(2)
Output (384, 192, 2)
```

## Analysis Features

### 1. Activation Threshold Analysis
- Threshold: 0.1
- Analyzes which neurons activate above threshold
- Creates bar plots for 4 temporal segments

### 2. Sparsity vs Reconstruction Error
- Tests neuron importance by masking
- Two approaches:
  - **Top-t active neurons**: Based on activation frequency
  - **Top-k magnitude neurons**: Based on mean activation magnitude
- Compares reconstruction with top neurons vs complement neurons

### 3. Single Snapshot Reconstruction
- Selects specific snapshot (default: 67)
- Shows three reconstructions:
  - Original snapshot
  - Using only top-k active neurons
  - Using only complement (suppressed) neurons
- Displays L2 reconstruction error for each

## Expected Outputs

### Plots Generated

1. **Velocity Magnitude Visualization**: Original snapshot velocity field
2. **Temporal Activation Plots**: 4 subplots + 1 full dataset plot
3. **Sparsity Curves**: Error vs sparsity percentage (two methods)
4. **Reconstruction Comparison**: 3-panel comparison with error metrics

### Performance Metrics

- Training/validation loss curves
- Normalized L2 reconstruction error
- Neuron activation statistics

## Troubleshooting

### Common Issues

**Out of Memory Error**
```python
# Reduce batch size
dense_ae.fit(..., batch_size=16, ...)  # Instead of 32
```

**File Not Found Error**
- Check data file paths
- Ensure data directory exists
- Verify file permissions

**GPU Not Detected**
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Slow Training**
- Enable GPU acceleration
- Reduce number of epochs for testing
- Use smaller dataset initially

## Customization

### Modify Latent Dimension
```python
# Change from 72 to desired size
encoded_latent = Dense(NEW_SIZE, activation=act, name='latent_dense')
x = Dense(NEW_SIZE, activation=act)(encoded_latent)
```

### Adjust Sparsity Analysis Parameters
```python
threshold = 0.1      # Activation threshold
top_n = 7            # Number of top neurons to analyze
step = 4             # Step size for sparsity loop
snapshot = 67        # Snapshot index for visualization
```

### Change Training Parameters
```python
dense_ae.fit(
    X_train, X_train,
    epochs=50,           # Increase/decrease epochs
    batch_size=64,       # Adjust batch size
    shuffle=True,
    validation_data=(X_test, X_test)
)
```

## Citation

If you use this code in your research, please cite appropriately.

## License

[Specify your license here]

## Contact

[Your contact information]

## Acknowledgments

- TensorFlow/Keras for deep learning framework
- Project contributors and data sources
