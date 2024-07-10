# Human Detection using Deep Learning

## Project Overview
This project focuses on building and evaluating deep learning models for human detection using image data. The models implemented include a custom CNN (Convolutional Neural Network), VGG16, Xception, and ResNet50. The project involves data loading and preprocessing, model training, evaluation, and visualization of the results.

## Dataset
The dataset used in this project is the Human Detection Dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/constantinwerner/human-detection-dataset).

## Requirements
- Python 3.6+
- Jupyter Notebook or Google Colab
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`, `tensorflow`, `keras`, `opencv-python`

## Project Structure
```
.
├── data/
│   ├── human-detection-dataset/
│   └── ...
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── cnn_model.ipynb
│   ├── vgg16_model.ipynb
│   ├── xception_model.ipynb
│   └── resnet_model.ipynb
├── models/
│   ├── cnn_model.h5
│   ├── vgg16_model.h5
│   ├── xception_model.h5
│   └── resnet_model.h5
└── README.md
```

## Setup and Usage
### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/human-detection.git
cd human-detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download and Prepare the Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/constantinwerner/human-detection-dataset) and place it in the `data/` directory.

### Step 4: Data Preprocessing
Run the `data_preprocessing.ipynb` notebook to load, preprocess, and split the dataset into training and testing sets.

### Step 5: Train the Models
You can train different models by running the corresponding notebooks:
- `cnn_model.ipynb`
- `vgg16_model.ipynb`
- `xception_model.ipynb`
- `resnet_model.ipynb`

### Step 6: Evaluate the Models
Each notebook includes code to evaluate the models using metrics such as accuracy, F1 score, ROC AUC, and confusion matrix. The evaluation results are visualized using plots.

## Results
The results include:
- Training and validation loss and accuracy plots
- ROC AUC curves
- Confusion matrices
- Classification reports
- Accuracy and F1 scores

## Model Comparison
| Model     | Accuracy | F1 Score | ROC AUC | 
|-----------|----------|----------|---------|
| CNN       | 62.34%   | 74.34    | 0.68   |
| VGG16     | 83.98%   | 86.93    | 0.92   |
| Xception  | 85.28%   | 87.31    | 0.94   |
| ResNet50  | 71.43%   | 79.11    | 0.82   |

## Conclusion
This project demonstrates the application of various deep learning models for human detection. It provides a comprehensive comparison of different models' performance and highlights the effectiveness of transfer learning using pre-trained models like VGG16, Xception, and ResNet50.

## Future Work
- Experiment with other architectures like InceptionV3, DenseNet, etc.
- Perform hyperparameter tuning for better model performance.
- Deploy the best-performing model as a web service.

## References
- [Kaggle Human Detection Dataset](https://www.kaggle.com/constantinwerner/human-detection-dataset)
- [TensorFlow Keras Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
