# Multilingual BERT-based CNN/LSTM Models

This repository contains the source code for training and evaluating Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) models on top of Multilingual BERT (mBERT) features. The models are tested on the task of Stance Detection, where the goal is to automatically classify the stance of a given article towards a given claim as agree, disagree, discuss, or unrelated.

## Models:

* **Convolutional Neural Network (CNN)**
* **Bidirectional Long Short-Term Memory (BiLSTM)**
* **CNN-BiLSTM**
* **BiLSTM-CNN**
* **Ensemble Model (CNN + BiLSTM)**

## Experiments:

### Dataset:
The experiments were conducted on Google Colab using a AraStance (Alhindi et al., [2021](https://aclanthology.org/2021.nlp4if-1.9/)) dataset.

### Experimental Settings:

Two experimental settings were explored:

1. **Last Layer Features**: The features from the final layer of mBERT was used as input to the CNN and LSTM models.
2. **Last Four Layers Features**: The features from the last four layers of mBERT were concatenated and used as input to the CNN and LSTM models.

### Evaluation Metrics:

* **Accuracy:** the ratio of the number of correct predictions to the total number of predictions.
* **F1-Score:** the harmonic mean of precision and recall.
* **Macro F1-score:** average of per-class f1-scores.

## Results and Analysis:

The results of each set of experiments are presented in the folders above.

## Future Work:

1. **Multiple Runs:** To obtain more robust results, consider running multiple training sessions with different random seeds and averaging the performance across the runs.
2. **Explore other multilingual models:** Experiment with other state-of-the-art multilingual models.
3. **Investigate data augmentation techniques:** Explore techniques to improve data diversity and model robustness.
4. **Fine-tune on larger and more diverse datasets:** Train the models on larger and more diverse datasets to enhance their generalizability.
 
In addition, it is possible to further improve the performance of the models on this classification task by carefully considering the following:

5. **Class Imbalance**: Techniques like class weighting or oversampling could be explored to address the class imbalance in the dataset.
6. **Hyperparameter Tuning**: Conduct a thorough hyperparameter search to optimize the performance of each model.

## Software/Libraries:

- Python 3.10.12
- NumPy 1.26.4
- PyTorch 2.5.1+cu121
- Transformers 4.46.3
- Scikit-learn 1.5.2