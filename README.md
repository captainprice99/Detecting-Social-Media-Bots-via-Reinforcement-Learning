# BotCatcher: Social Media Bot Detection using Graph Neural Networks and Reinforcement Learning

## Overview
This project implements a sophisticated bot detection system for social media platforms using Graph Neural Networks (GNNs) and Reinforcement Learning (RL). The system analyzes user behavior patterns and network structures to distinguish between genuine accounts and automated bots with high accuracy.

## Features
- **High Accuracy**: Achieves 98.35% accuracy in bot detection
- **Precision & Recall**: 99.32% precision and 98.06% recall
- **Graph-Based Analysis**: Utilizes k-Nearest Neighbors (kNN) similarity graphs to model user relationships
- **Multi-Feature Analysis**: Considers various user metadata and interaction patterns
- **Scalable**: Designed to handle large-scale social network data

## Tech Stack
- **Python 3.9+**
- **PyTorch Geometric** - For graph neural network implementation
- **Pandas & NumPy** - For data manipulation and processing
- **scikit-learn** - For model evaluation and metrics
- **Matplotlib** - For data visualization

## Dataset
This project uses the following datasets:
- **Cresci-2017**: Contains labeled genuine and bot accounts from Twitter
- **TwiBot-22**: Additional dataset for model training and validation

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Detecting-Social-Media-Bots-via-Reinforcement-Learning.git
   cd Detecting-Social-Media-Bots-via-Reinforcement-Learning
   ```

2. **Set up a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install torch torch-geometric pandas numpy scikit-learn matplotlib
   ```

## Project Structure
```
.
├── cresci-2017/               # Dataset directory
│   ├── genuine_accounts/      # Genuine user accounts data
│   ├── social_spambots_1/     # Social spambots dataset 1
│   ├── social_spambots_2/     # Social spambots dataset 2
│   ├── social_spambots_3/     # Social spambots dataset 3
│   └── traditional_spambots_1/ # Traditional spambots dataset
├── gnnRL.ipynb               # Main Jupyter notebook with implementation
├── README.md                 # This file
└── requirements.txt          # Project dependencies
```

## Usage

1. **Data Preparation**
   - Ensure the Cresci-2017 dataset is downloaded and placed in the `cresci-2017` directory
   - The dataset can be obtained from: [Bot Repository](https://botometer.osome.iu.edu/bot-repository/datasets.html)

2. **Running the Model**
   - Open and run the `gnnRL.ipynb` Jupyter notebook
   - The notebook is organized into logical sections:
     1. Environment setup and dependencies
     2. Data loading and preprocessing
     3. Graph construction using kNN similarity
     4. Model architecture definition
     5. Training loop
     6. Evaluation and visualization

## Methodology

1. **Graph Construction**
   - Users are represented as nodes in a graph
   - Edges are created based on k-Nearest Neighbors (kNN) similarity of user features
   - Node features include user metadata and activity statistics

2. **Model Architecture**
   - Graph Neural Network (GNN) for learning node representations
   - Reinforcement Learning agent for decision making
   - Attention mechanisms for focusing on important user interactions

3. **Training Process**
   - Supervised learning with labeled bot/human accounts
   - Reinforcement learning for optimizing detection policies
   - Cross-validation for robust performance evaluation

## Results

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 98.35%  |
| Precision | 99.32%  |
| Recall    | 98.06%  |
| F1-Score  | 98.68%  |

## Future Improvements

- **Integration with Real-time APIs**: Connect to Twitter/Facebook APIs for live bot detection
- **Enhanced Feature Engineering**: Incorporate more sophisticated behavioral features
- **Semi-supervised Learning**: Leverage unlabeled data for improved generalization
- **Multi-platform Support**: Extend to other social media platforms
- **Explainability**: Add model interpretation tools to understand detection decisions
- **Federated Learning**: Enable privacy-preserving model training across multiple instances

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The Cresci-2017 dataset provided by the Bot Repository
- PyTorch Geometric team for the GNN framework
- Open-source community for various libraries and tools used in this project