## Extractive Question Answering 

In this project, I fine-tuned a customized version of the famous persian pre-trained language model [ParsBert](https://github.com/hooshvare/parsbert).

The dataset I used is [PQuAD](https://github.com/AUT-NLP/PQuAD) which is a crowd-sourced reading comprehension dataset on Persian Language. It includes 80,000 questions along with their answers, with 25% of the questions being unanswerable.
## Architecture
We added two feed-forward layers on top of last layer of ParsBert.\
You may ask Why two layers?\
Since we have embeddings for each token in last layer, we add one layer for predicting each token being the start token of the answer and the other layer is responsible for predicting each token being the end token of the answer. Weights of these two layers will be optimized during fine-tuning process. 

![68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f6d61782f313834302f312a51684958734442456e414e4c584d4130794f4e7878412e706e67](https://github.com/user-attachments/assets/1ea14a75-fec7-4690-a57f-4aeb27af8a99)

## Hyperparameters
I set batch_size to 32 due to the limitations of GPU memory in Google Colab.

```
batch_size = 32,
n_epochs = 2,
max_seq_len = 256,
learning_rate = 5e-5
```
## Results
Through mentioned settings we got the results below:\
Exact Match: 47.44\
F1-Score: 81.96
## How to use

### Pytorch
```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
path = 'pedramyazdipoor/parsbert_question_answering_PQuAD'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForQuestionAnswering.from_pretrained(path)
```

### Tensorflow
```python
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
path = 'pedramyazdipoor/parsbert_question_answering_PQuAD'
tokenizer = AutoTokenizer.from_pretrained(path)
model = TFAutoModelForQuestionAnswering.from_pretrained(path)
```

### Inference for pytorch
I leave Inference for tensorflow as an excercise for you :) .

There are some considerations for inference:
1) Start index of answer must be smaller than end index.
2) The span of answer must be within the context.
3) The selected span must be the most probable choice among N pairs of candidates.

```python
def generate_indexes(start_logits, end_logits, N, max_index):
  
  output_start = start_logits
  output_end = end_logits

  start_indexes = np.arange(len(start_logits))
  start_probs = output_start
  list_start = dict(zip(start_indexes, start_probs.tolist()))
  end_indexes = np.arange(len(end_logits))
  end_probs = output_end
  list_end = dict(zip(end_indexes, end_probs.tolist()))

  sorted_start_list = sorted(list_start.items(), key=lambda x: x[1], reverse=True) #Descending sort by probability
  sorted_end_list = sorted(list_end.items(), key=lambda x: x[1], reverse=True)

  final_start_idx, final_end_idx = [[] for l in range(2)]

  start_idx, end_idx, prob = 0, 0, (start_probs.tolist()[0] + end_probs.tolist()[0])
  for a in range(0,N):
    for b in range(0,N):
      if (sorted_start_list[a][1] + sorted_end_list[b][1]) > prob :
        if (sorted_start_list[a][0] <= sorted_end_list[b][0]) and (sorted_end_list[a][0] < max_index) :
          prob = sorted_start_list[a][1] + sorted_end_list[b][1]
          start_idx = sorted_start_list[a][0]
          end_idx = sorted_end_list[b][0]
  final_start_idx.append(start_idx)    
  final_end_idx.append(end_idx)      

  return final_start_idx[0], final_end_idx[0]
```

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval().to(device)
text = 'سلام من پدرامم 26 سالمه'
question = 'چند سالمه؟'
encoding = tokenizer(text,question,add_special_tokens = True,
                     return_token_type_ids = True,
                     return_tensors = 'pt',
                     padding = True,
                     return_offsets_mapping = True,
                     truncation = 'only_first',
                     max_length = 32)
out = model(encoding['input_ids'].to(device),encoding['attention_mask'].to(device), encoding['token_type_ids'].to(device))
#we had to change some pieces of code to make it compatible with one answer generation at a time
#If you have unanswerable questions, use out['start_logits'][0][0:] and out['end_logits'][0][0:] because <s> (the 1st token) is for this situation and must be compared with other tokens.
#you can initialize max_index in generate_indexes() to put force on tokens being chosen to be within the context(end index must be less than seperator token).
answer_start_index, answer_end_index = generate_indexes(out['start_logits'][0][1:], out['end_logits'][0][1:], 5, 0)
print(tokenizer.tokenize(text + question))
print(tokenizer.tokenize(text + question)[answer_start_index : (answer_end_index + 1)])
>>> ['▁سلام', '▁من', '▁پدر', 'ام', 'م', '▁26', '▁سالم', 'ه', 'نام', 'م', '▁چیست', '؟']
>>> ['▁26']
```

## Acknowledgments
We did this project thanks to the fantastic job done by [HooshvareLab](https://huggingface.co/HooshvareLab/bert-fa-base-uncased).
We also express our gratitude to PQuAD team for facilitating dataset gathering.

## Contributors
- Pedram Yazdipoor : [Linkedin](https://www.linkedin.com/in/pedram-yazdipour/)
