# AI Visual Lab 🤖

An interactive AI learning platform built with Streamlit that helps you understand artificial intelligence concepts through visualizations and hands-on examples.

## Features

- **Data Basics**: Understanding data structures and preprocessing
- **Regression**: Linear and polynomial regression models
- **Classification**: Binary and multi-class classification
- **Neural Networks**: Deep learning fundamentals with architecture visualization
- **CNN Visuals**: Convolutional Neural Networks and convolution operations
- **GenAI Tokenization**: How language models process text
- **GenAI Attention**: Attention mechanisms in transformers
- **GenAI Generation**: Text generation techniques and sampling strategies
- **Capstone**: Real-world project ideas and guidance

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ai-visual-lab.git
cd ai-visual-lab
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

### Docker Setup

1. Build the Docker image:
```bash
docker build -t ai-visual-lab .
```

2. Run the container:
```bash
docker run -p 8501:8501 ai-visual-lab
```

3. Access the app at `http://localhost:8501`

## Project Structure

```
ai-visual-lab/
│
├── app.py                      # Main application with navigation
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── README.md                   # Project documentation
│
├── modules/                    # Educational modules
│   ├── welcome.py             # Welcome page
│   ├── intro.py               # Introduction to AI
│   ├── data_basics.py         # Data fundamentals
│   ├── regression.py          # Regression analysis
│   ├── classification.py      # Classification models
│   ├── neural_networks.py     # Neural network basics
│   ├── cnn_visuals.py         # CNN visualization
│   ├── genai_tokenization.py # Tokenization demo
│   ├── genai_attention.py     # Attention mechanism
│   ├── genai_generation.py    # Text generation
│   └── capstone.py            # Project ideas
│
└── .streamlit/
    └── config.toml            # Streamlit configuration
```

## Technologies Used

- **Streamlit**: Interactive web application framework
- **NumPy & Pandas**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms
- **TensorFlow & PyTorch**: Deep learning frameworks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

Built for educational purposes to make AI concepts more accessible and interactive.
