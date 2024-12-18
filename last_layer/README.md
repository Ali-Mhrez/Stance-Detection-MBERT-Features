# Results: Using Last Layer Features

In this experiment, we utilized the final layer of the pre-trained Multilingual BERT model to extract semantic and syntactic features from the input text. These features were then fed into Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architectures to perform Stance Detection.

The results obtained from this setting provide insights into the effectiveness of leveraging the high-level representations captured by the final layer of BERT.

## Results:

### Validation Results

| Model | Accuracy | Agree | Disagree | Discuss | Unrelated | Macro f1-score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| CNN | 0.819 | 0.833 | 0.774 | 0.500 | 0.874 | 0.745 |
| BiLSTM | 0.756 | 0.707 | 0.631 | 0.452 | 0.865 | 0.664 |
| CNN-BiLSTM | 0.773 | 0.754 | 0.667 | 0.500 | 0.849 | 0.693 |
| BiLSTM-CNN | 0.780 | 0.736 | 0.716 | 0.517 | 0.867 | 0.709 |
| CNN+BiLSTM | 0.801 | 0.771 | 0.745 | 0.492 | 0.882 | 0.722 |

### Testing Results

| Model | Accuracy | Agree | Disagree | Discuss | Unrelated | Macro f1-score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| CNN | 0.851 | 0.865 | 0.737 | 0.408 | 0.923 | 0.733 |
| BiLSTM | 0.816 | 0.788 | 0.707 | 0.435 | 0.906 | 0.709 |
| CNN-BiLSTM | 0.833 | 0.828 | 0.702 | 0.396 | 0.911 | 0.709 |
| BiLSTM-CNN | 0.837 | 0.791 | 0.705 | 0.466 | 0.937 | 0.725 |
| CNN+BiLSTM | 0.854 | 0.854 | 0.768 | 0.466 | 0.921 | 0.752 |