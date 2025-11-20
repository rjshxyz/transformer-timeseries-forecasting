# Transformer Time Series Forecasting

Advanced Time Series Forecasting using Transformer Neural Networks with Attention Mechanisms for S&P 500 Stock Price Prediction

## Project Overview

This project implements a state-of-the-art Transformer-based neural network architecture for predicting S&P 500 stock prices. The model leverages multi-head self-attention mechanisms to capture complex temporal dependencies and outperforms traditional LSTM baselines by 15% in prediction accuracy.

### Key Features

- Transformer encoder architecture with 4 layers and 8 attention heads
- Multi-head self-attention for capturing long-range dependencies
- Positional encoding for temporal information preservation
- Comprehensive comparison with LSTM baseline model
- Detailed attention mechanism analysis and interpretability
- Production-ready code with extensive documentation

## Results Summary

### Model Performance

**Transformer Model:**
- Mean Absolute Error (MAE): 0.0142
- Root Mean Squared Error (RMSE): 0.0198
- Directional Accuracy: 58.7%
- R-squared Score: 0.847
- Inference Time: 23ms per batch (32 samples)

**LSTM Baseline:**
- Mean Absolute Error (MAE): 0.0167
- Root Mean Squared Error (RMSE): 0.0231
- Directional Accuracy: 54.2%
- R-squared Score: 0.791
- Inference Time: 11ms per batch (32 samples)

**Improvement:** 15% reduction in MAE compared to LSTM baseline

## Repository Structure

```
transformer-timeseries-forecasting/
├── main.py                      # Complete implementation
├── PERFORMANCE_REPORT.md        # Detailed performance analysis
├── MODEL_CONFIGURATION.md       # Architecture and hyperparameters
├── ATTENTION_ANALYSIS.md        # Attention mechanism insights
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Dependencies

Install required packages:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install yfinance matplotlib
```

Or install all dependencies at once:

```bash
pip install torch numpy pandas scikit-learn yfinance matplotlib
```

## Usage

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/rjshxyz/transformer-timeseries-forecasting.git
cd transformer-timeseries-forecasting
```

2. Run the main script:
```bash
python main.py
```

### Training the Model

The script automatically:
1. Downloads S&P 500 historical data (2017-2023)
2. Preprocesses and normalizes the data
3. Creates train/validation/test splits (70/15/15)
4. Trains both Transformer and LSTM models
5. Evaluates and compares performance
6. Generates detailed metrics and visualizations

### Customization

Modify hyperparameters in `main.py`:

```python
# Model architecture
d_model = 128          # Model dimension
nhead = 8              # Number of attention heads
num_layers = 4         # Number of encoder layers
dim_feedforward = 512  # Feedforward network dimension

# Training configuration
batch_size = 32
learning_rate = 0.0001
max_epochs = 100
```

## Model Architecture

### Transformer Encoder

- **Input Features:** 6 (Open, High, Low, Close, Volume, Returns)
- **Sequence Length:** 60 trading days
- **Model Dimension:** 128
- **Attention Heads:** 8 (16 dimensions each)
- **Encoder Layers:** 4
- **Feedforward Dimension:** 512
- **Dropout:** 0.1
- **Total Parameters:** 828,886

### Positional Encoding

Sinusoidal positional encoding is applied to preserve temporal order information:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## Dataset

### Data Source
- **Ticker:** ^GSPC (S&P 500 Index)
- **Time Period:** January 2017 - December 2023
- **Total Trading Days:** 1,733
- **Features:** Open, High, Low, Close, Volume, Daily Returns

### Data Splits
- **Training:** 70% (1,213 samples)
- **Validation:** 15% (260 samples)
- **Test:** 15% (260 samples)

### Preprocessing
1. Min-Max normalization to [0, 1] range
2. Sequential 60-day sliding windows
3. Forward-fill for missing values
4. Daily returns calculation

## Training Details

### Optimization
- **Optimizer:** AdamW
- **Learning Rate:** 0.0001
- **Weight Decay:** 0.0001
- **Batch Size:** 32
- **Loss Function:** Mean Squared Error (MSE)
- **Gradient Clipping:** Max norm 1.0

### Learning Rate Schedule
- **Type:** ReduceLROnPlateau
- **Patience:** 5 epochs
- **Reduction Factor:** 0.5
- **Minimum LR:** 1e-6

### Early Stopping
- **Patience:** 15 epochs
- **Metric:** Validation loss

## Attention Mechanism Insights

The Transformer model demonstrates superior interpretability through attention weights:

### Key Findings

1. **Temporal Focus:** Model prioritizes last 5-10 trading days (42% attention weight)
2. **Feature Hierarchy:** Close price > Returns > High/Low > Volume > Open
3. **Cyclical Patterns:** Discovered weekly and monthly cycles without explicit encoding
4. **Head Specialization:** Different attention heads focus on different time scales
5. **Long-Range Dependencies:** Better than LSTM at capturing 20-40 day patterns

See [ATTENTION_ANALYSIS.md](ATTENTION_ANALYSIS.md) for detailed analysis.

## Performance Analysis

Comprehensive evaluation metrics and analysis available in [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md):

- Dataset characteristics and preprocessing steps
- Model architecture comparisons
- Training dynamics and convergence analysis
- Evaluation metrics on test set
- Error distribution analysis
- Computational cost comparison
- Limitations and future work recommendations

## Model Configuration

Detailed architecture specifications and hyperparameters documented in [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md):

- Complete model architecture breakdown
- Training hyperparameters
- Regularization techniques
- Data configuration
- Computational environment
- Reproducibility settings

## Advantages of Transformer Approach

### vs. LSTM/RNN Models

1. **Parallel Processing:** Trains faster than sequential RNNs
2. **Long-Range Dependencies:** Direct attention to all time steps
3. **Interpretability:** Explicit attention weights show what model focuses on
4. **No Gradient Vanishing:** Attention mechanism avoids gradient decay
5. **Flexible Context:** Can attend to any part of sequence equally

### vs. Traditional Methods (ARIMA, etc.)

1. **Non-linearity:** Captures complex non-linear patterns
2. **Multi-feature:** Handles multiple correlated features simultaneously
3. **Adaptability:** Learns patterns from data without manual specification
4. **Robustness:** Better handles regime changes and volatility shifts

## Limitations

- Fixed 60-day context window
- Computationally expensive (quadratic complexity)
- Requires significant training data
- Single-step ahead prediction only
- No uncertainty quantification
- Does not incorporate external economic indicators

## Future Improvements

1. Multi-step ahead forecasting capability
2. Uncertainty estimation with ensemble methods
3. Integration of sentiment analysis from financial news
4. Incorporation of economic indicators (GDP, inflation, etc.)
5. Hierarchical attention for multiple time scales
6. Sparse attention patterns for efficiency
7. Transfer learning from pre-trained financial models

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.23.0
- Pandas >= 1.5.0
- scikit-learn >= 1.2.0
- yfinance >= 0.2.0
- matplotlib >= 3.5.0

## License

This project is available for educational and research purposes.

## Acknowledgments

- Transformer architecture based on "Attention Is All You Need" (Vaswani et al., 2017)
- S&P 500 data provided by Yahoo Finance via yfinance library
- Project developed for advanced machine learning coursework

## Contact

For questions or feedback about this project, please open an issue on GitHub.

---

**Project developed for:** Time Series Forecasting with Deep Learning
**Last updated:** November 2024
