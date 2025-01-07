## Extractive Question Answering 

In this project, I fine-tuned a customized version of the famous persian pre-trained language model [ParsBert](https://github.com/hooshvare/parsbert).

The dataset I used is [PQuAD](https://github.com/AUT-NLP/PQuAD) which is a crowd-sourced reading comprehension dataset on Persian Language. It includes 80,000 questions along with their answers, with 25% of the questions being unanswerable.

## Hyperparameters
I set batch_size to 32 due to the limitations of GPU memory in Google Colab.

```
batch_size = 32,
n_epochs = 2,
max_seq_len = 256,
learning_rate = 5e-5
``` 
Through mentioned settings we got the results below:\
Exact Match: 47.44
F1-Score: 81.96
