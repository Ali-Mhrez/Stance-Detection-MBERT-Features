# Repository of the 7th chapter of my dissertation: [Research Proposal](https://github.com/Ali-Mhrez/Dissertation/blob/main/dissertation.pdf), pages 93--108

This repository contains the code for a Ph.D. research project that focuses on improving the performance of mBERT for fake news stance detection. The core task is a multi-class classification problem where a news article's body text is classified with respect to its headline into one of four categories: agree, disagree, discuss, or unrelated. This work explores specific methodologies to enhance mBERT's effectiveness on this challenging task.

## Goal and Background:

The shift from traditional word embedding techniques (like CBOW and Skip-Gram) to transformer-based models marks a pivotal advancement in NLP. This superiority stems from the self-attention mechanism, which allows transformers to generate richer, more contextualized vector representations than their predecessors. The effectiveness of these models is rooted in transfer learning, where pre-trained knowledge from a large corpus is adapted to a specific downstream task.

However, a key challenge remains: simply fine-tuning a transformer model like mBERT (Multilingual BERT) using its default methodology may not be the most optimal approach for a complex task like fake news stance detection. The goal of this research is to investigate and propose alternative methodologies that leverage mBERT's powerful representation vectors while enriching them with additional processing.

This study makes the following specific contributions:
1. **Novel Architectures:** We propose and evaluate several deep learning architectures that combine mBERT's contextual embeddings with additional processing layers to enhance its performance on Arabic fake news stance detection.
2. **Improved Methodology:** We introduce a new methodology for using mBERT that surpasses the effectiveness of the standard fine-tuning approach.
3. **Hybrid Model Evaluation:** We test the effectiveness of hybrid models by using features extracted from mBERT as input to traditional deep learning models, including CNNs and BiLSTMs, a configuration that has not been previously explored for this task.

## Dataset
The [AraStance](https://aclanthology.org/2021.nlp4if-1.9/) dataset includes article bodies, headlines, and a corresponding class label. The label indicates the stance of the article body with respect to the headline. The article body can either Agree (AGR) or Disagree (DSG) with the headline, it can Discuss (DSC) it or be completely Unrelated (UNR).
| Data Source | Data Type | Instances | AGR | DSG | DSC | UNR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| [repo](https://github.com/Tariq60/arastance) | News articles | 4,063 | 25.1% | 11.0% | 9.5% | 54.3% |

## Data Preprocessing
AraStance is already divided into: Training, Validation, Testing sets.  
No Special preprocessing was conducted except tokenization using the default tokenizer of each model.

## Models

<table>
      <thead>
            <th>Model</th>
            <th colspan=2>Layers</th>
      </thead>
      <tbody>
            <tr>
                  <td rowspan=5>CNN</td>
                  <td colspan=2>
                        Conv1D (filters=100, kernel_size=2, activation=relu, use_bias=True)<br>
                        Conv1D (filters=100, kernel_size=3, activation=relu, use_bias=True)<br>
                        Conv1D (filters=100, kernel_size=4, activation=relu, use_bias=True)
                  </td>
            </tr>
            <tr>
                  <td colspan=2>
                        GlobalMaxPooling1D<br>
                        GlobalMaxPooling1D<br>
                        GlobalMaxPooling1D
                  </td>
            </tr>
            <tr>
                  <td colspan=2>Concatenation (axis=1)</td>
            </tr>
            <tr>
                  <td colspan=2>Dropout (rate=0.5)</td>
            </tr>
            <tr>
                  <td colspan=2>Dense (units=4)</td>
            </tr>
            <tr>
                  <td rowspan=3>BiLSTM</td>
                  <td colspan=2>Bidirectional( LSTM (units=32, activation=tanh, return_sequences=True) ) </td>
            </tr>
            <tr>
                  <td colspan=2>Bidirectional( LSTM (units=32, activation=tanh, return_sequences=False) ) </td>
            </tr>
            <tr>
                  <td colspan=2>Dense (units=4)</td>
            </tr>
            <tr>
                  <td rowspan=5>CNN-BiLSTM</td>
                  <td colspan=2>
                        Conv1D (filters=100, kernel_size=2, activation=relu, use_bias=True)<br>
                        Conv1D (filters=100, kernel_size=3, activation=relu, use_bias=True)<br>
                        Conv1D (filters=100, kernel_size=4, activation=relu, use_bias=True)
                  </td>
            </tr>
            <tr>
                  <td colspan=2>Concatenation (axis=2)</td>
            </tr>
            <tr>
                  <td colspan=2>Bidirectional( LSTM (units=32, activation=tanh, return_sequences=False) ) </td>
            </tr>
            <tr>
                  <td colspan=2>Dropout (rate=0.2)</td>
            </tr>
            <tr>
                  <td colspan=2>Dense (units=4)</td>
            </tr>
            <tr>
                  <td rowspan=7>BiLSTM-CNN</td>
                  <td colspan=2>Bidirectional( LSTM (units=32, activation=tanh, return_sequences=True) ) </td>
            </tr>
            <tr>
                  <td colspan=2>Bidirectional( LSTM (units=32, activation=tanh, return_sequences=True) ) </td>
            </tr>
            <tr>
                  <td colspan=2>
                        Conv1D (filters=50, kernel_size=2, activation=relu, use_bias=True)<br>
                        Conv1D (filters=50, kernel_size=3, activation=relu, use_bias=True)<br>
                        Conv1D (filters=50, kernel_size=4, activation=relu, use_bias=True)
                  </td>
            </tr>
            <tr>
                  <td colspan=2>
                        GlobalMaxPooling1D<br>
                        GlobalMaxPooling1D<br>
                        GlobalMaxPooling1D
                  </td>
            </tr>
            <tr>
                  <td colspan=2>Concatenation (axis=1)</td>
            </tr>
            <tr>
                  <td colspan=2>Dropout (rate=0.2)</td>
            </tr>
            <tr>
                  <td colspan=2>Dense (units=4)</td>
            </tr>
            <tr>
                  <td rowspan=3>Ensemble (CNN,BiLSTM</td>
                  <td>CNN</td>
                  <td>BiLSTM</td>
            </tr>
            <tr>
                  <td colspan=2>Add Logits</td>
            </tr>
            <tr>
                  <td colspan=2>Dense (units=4)</td>
            </tr>
      </tbody>
</table>

## Experimental Settings

Two experimental settings were explored:
1. **Last Layer Features**: The features from the final layer of mBERT was used as input to the CNN and LSTM models.
2. **Last Four Layers Features**: The features from the last four layers of mBERT were concatenated and used as input to the CNN and LSTM models.

## Hyperparameters

<table>
 <tr>
  <th rowspan=2>Model</th>
  <th rowspan=2>Sequence Length</th>
  <th rowspan=2>Batch Size</th>
  <th rowspan=2>Learning Rate</th>
  <th colspan=2>Epochs</th>
 </tr>
 <tr>
    <th>Last Layer</th>
  <th>Last Four Layers</th>
 </tr>
 <tr>
  <td>CNN</td><td>256</td><td>64</td><td>1e-4</td><td>55</td><td>46</td>
 </tr>
  <tr>
  <td>BiLSTM</td><td>256</td><td>64</td><td>1e-2</td><td>50</td><td>50</td>
 </tr>
  <tr>
  <td>CNN-BiLSTM</td><td>256</td><td>64</td><td>1e-4</td><td>37</td><td>33</td>
 </tr>
  <tr>
  <td>BiLSTM-CNN</td><td>256</td><td>64</td><td>1e-4</td><td>55</td><td>44</td>
 </tr>
  <tr>
  <td>CNN~BiLSTM</td><td>256</td><td>64</td><td>1e-4</td><td>45</td><td>44</td>
 </tr>
</table>

## Key Results
Our research focused on the BERT-ESDM methodology to improve mBERT's performance for stance detection. The key findings are as follows:
1. **Improved Performance with CNN Filters:** Our approach of using a Convolutional Neural Network (CNN) to filter the contextual vectors from mBERT's final layer resulted in a significant performance improvement of 1.71% (F1-macro score).
2. **Leveraging Deeper Layers:** Performance saw an even greater boost—an improvement of 2.16% (F1-macro score)—when the CNN filtered contextual vectors extracted from the last four layers of the mBERT transformer, indicating that deeper layers contain richer information.
3. **Sequential Processing is Not Required:** The contextual vectors produced by mBERT are so information-rich that they do not require the sequential processing typically performed by a BiLSTM model for this task.
4. **Effectiveness of the Proposed Methodology:** The proposed BERT-ESDM methodology offers a highly effective alternative to standard fine-tuning, providing a novel and superior approach for adapting mBERT to specific datasets.
5. **Information Content:** The last four layers of the mBERT transformer provide a significantly greater amount of information compared to just the last layer, which is crucial for achieving state-of-the-art results.

## Requirements

- Python 3.10.12
- NumPy 1.26.4
- PyTorch 2.5.1+cu121
- Transformers 4.46.3
- Scikit-learn 1.5.2

## Citation
```bash
@incollection{amhrez-esdm,
author = {Mhrez, ali; Ramadan, Wassim; Abo Saleh, Naser},
title = {Research Proposal: ESDM Methodology},
booktitle = {Stance Detection in Natural Language Texts Using Deep Learning Techniques},
publisher = {University of Homs},
chapter = {5},
pages = {93--108},
year = {2024},
url = {https://github.com/Ali-Mhrez/Dissertation},
note = {This chapter is based on a dissertation submitted in partial fulfillment of the requirements for the degree of Doctor of Philosophy.}
}
```
