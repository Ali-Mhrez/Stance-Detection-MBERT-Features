# Results: Using Last Four Layer Features

In this experiment, we explored a deeper level of representation by concatenating the features from the last four layers of the pre-trained Multilingual BERT model. This approach aimed to capture a wider range of contextual information, from fine-grained details to broader semantic meanings. The resulting feature vectors were then fed into Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architectures for Stance Detection.

By leveraging a richer representation, we investigate the potential for improved performance in capturing complex linguistic nuances and identifying subtle stance cues.

## Results:

### Validation Results

| Model | Accuracy | Agree | Disagree | Discuss | Unrelated | Macro f1-score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| CNN | **0.831** | **0.864** | 0.735 | 0.505 | **0.896** | **0.750** |
| BiLSTM | 0.793 | 0.789 | **0.736** | 0.432 | 0.875 | 0.708 |
| CNN-BiLSTM | 0.796 | 0.791 | 0.651 | 0.458 | 0.877 | 0.694 |
| BiLSTM-CNN | 0.798 | 0.768 | 0.677 | 0.512 | 0.895 | 0.713 |
| CNN+BiLSTM | 0.822 | 0.820 | 0.709 | **0.538** | **0.896** | 0.741 |

### Testing Results

| Model | Accuracy | Agree | Disagree | Discuss | Unrelated | Macro f1-score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| CNN | **0.861** | **0.850** | **0.780** | **0.451** | **0.934** | **0.754** |
| BiLSTM | 0.808 | 0.783 | 0.714 | 0.392 | 0.894 | 0.696 |
| CNN-BiLSTM | 0.814 | 0.804 | 0.642 | 0.276 | 0.904 | 0.656 |
| BiLSTM-CNN | 0.831 | 0.833 | 0.673 | 0.431 | 0.918 | 0.714 |
| CNN+BiLSTM | 0.844 | 0.814 | 0.707 | 0.440 | 0.931 | 0.723 |