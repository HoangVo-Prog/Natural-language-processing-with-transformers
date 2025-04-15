2025-04-15 10:53

Status: [[Transformers]] [[Applications]]

Tags: [[Natural-language-processing-with-transformers-revised-edition.pdf]]  [[Text Classification]]

Pages: 21 - 55

# GOALS
You are a data scientist who needs to build a system that can automatically identify emotional states.

## TODO list:
- [x] Using variant of BERT called DistilBERT
  > [!toggle] Why using DistilBERT?
  > 
  > The main advantage of this model is that is achieves comparable performance to BERT, while significantly smaller and more efficient.
- [x] Train a classifier.
  > [!toggle] Classifier of what?
  > 
  > Classify the emotional states such as "anger" or "joy" that people express about your company's product.
- [x] Change the checkpoint of the pretrained model.
  > [!toggle] What is a checkpoint?
  > 
  > A checkpoint corresponds to the set of weights that are loaded into a given transformer architecture.

## FINAL GOAL
- [x] Given a tweet, train a model that can classify it into one of these emotions
- [x] Implement customize code
# NOTE

## The Dataset

- The datasets is about English Twitter messages.
- Contains 6 basics emotions: anger, disgust, fear, joy, sadness and surprise
### A First Look at Hugging Face Datasets

>Load dataset

```python
from datasets import load_dataset

emotions = load_dataset("dair-ai/emotion")
```
> Load data as dataframe

Option 1:
```python
import pandas as pd
from datasets import Dataset

dataset_url = "https://huggingface.co/datasets/transformersbook/emotion-train-split/raw/main/train.txt"

!wget {dataset_url} # get the dataset to local

!head -n 1 train.txt # print out the first line of the dataset

df = pd.read_csv("train.txt", sep=";", names=["text", "label"])

emotions_local = Dataset.from_pandas(df)
```

```bash
Dataset({
    features: ['text', 'label'],
    num_rows: 16000
})
```

Option 2:
```python
import pandas as pd
emtions.set_format(type="pandas")
df = emotions["train"][:]
df.head()
```

```bash
|text|label|
|---|---|
|0|i didnt feel humiliated|0|
|1|i can go from feeling so hopeless to so damned...|0|
|2|im grabbing a minute to post i feel greedy wrong|3|
|3|i am ever feeling nostalgic about the fireplac...|2|
|4|i am feeling grouchy|3|
```

### Looking at the Class Distribution
![[Pasted image 20250415123558.png]]
>
>The dataset is heavily imbalanced, in which, the "joy" and "sadness" classes appear frequently, while "love" and "surprise" are about 5-10 times rarer.

There are several ways to deal with imbalanced data:
- Randomly oversample the minority class.
- Randomly undersample the majority class.
- Gather more labeled data from the underrepresented classes.
Learn more at: [Imbalanced-learn library](https://imbalanced-learn.org/stable/)


### How Long Are Our Tweets?

  > [!toggle] maximum context size
  > 
  > Transformer models have a maximum input sequence length that is referred to as the maximum context size.

As DistilBERT, the maximum context size is 512 tokens. Below is about our tweets:

![[Pasted image 20250415124537.png]]



## From Text to Tokens

### Character Tokenization
```python
import torch
from torch.nn.functional import F


text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
input_ids = [token2idx[token] for token in tokenized_text]
input_ids = torch.tensor(input_ids)

one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))

one_hot_encodings.shape
```

```bash
torch.Size([38, 20])
```


### Word Tokenization
Simply split the corpus by whitespace 
=> Potential problem, punctuations will be eliminated.

Having a large vocabulary is a problem because it requires neural networks to have an enormous number of parameters. 
=> A common approach is to limit the vocabulary and discard rare words by classified as "unknown" and mapped to a shared "UNK" token


### Subword Tokenization (Character + Word)

>WordPiece, by BERT and DistilBERT tokenizers

```python
from transformers import AutoTokenizer

model_ckpt = "distilbert-base-cased" # same, be remain case-sensitivity
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
```

### Tokenizing the Whole Dataset

```python
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True) # Load from the pretrain model

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print(emotions_encoded["train"].column_names)
```

```bash
['attention_mask', 'input_ids', 'label', 'text']
```




## Training a Text Classifier

  > [!toggle] Vocabulary Size
  > 
  > **What it is**: The number of unique tokens (words, subwords, punctuation, etc.) the model can recognize.
  >  **Why it matters**: A larger vocabulary means the model can directly represent more words, which reduces the need for breaking rare words into subwords. However, larger vocabularies increase memory usage.
    Example: DistilBERT has a vocab size of 30,522.


  > [!toggle] Token Embedding Size
  > 
  >**What it is**: The size (in dimensions) of the vector used to represent each token in the input.
>  **Why it matters**: This is the first layer of the model — it maps each token to a continuous vector space. Larger sizes may capture richer information but increase model size and compute.
    Example:  In DistilBERT, the embedding size is typically **768** (same as the hidden size).

  > [!toggle] Hidden Size
  > 
  >**What it is**: The dimensionality of the hidden states inside the transformer layers — it defines how much information can be stored at each layer.
> **Why it matters**: This affects the model’s overall capacity and performance. Larger hidden sizes typically mean more expressive models, but also more memory and compute cost.vector space. Larger sizes may capture richer information but increase model size and compute.
    Example:  DistilBERT has a hidden size of 768, while `bert-large` has 1024.

We have two options to train such a model on our Twitter dataset:
- Feature extraction: We use the hidden states as features and just train a classifier on them, without modifying the pretrained model. 
- Fine-tuning: We train the whole model end-to-end, which also updates the parameters of the pretrained model. In the following sections we explore both options for DistilBERT and examine their trade-offs.

### Transformer as Feature Extractors


Freeze body's weights during training and use hidden states as features for the classifier. 
- Advantage: Quickly train a small or shallow model, such a model could be a neural classification, or a method like a random forest, which is not rely on gradients.

#### Using pretrained models

```python
from transformers import AutoModel


model_ckpt = "distilbert-base-cased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
```
The `AutoModel` class convert the token encodings to embeddings, and then feeds them through the encoder stack to return the hidden states. 

#### Extracting the last hidden states

> Encode string and convert the tokens to PyTorch tensors
```python
text = "this is a text"
inputs = tokenizer(text, return_tensors="pt") # pt is short for PyTorch
print(f"Input tensor shape: {inputs['input_ids'].size()}")
```

```bash
Input tensor shape: torch.Size([1, 6])
```

> Get the hidden states

```python
def extract_hidden_states(batch):
	# Place model inputs on the GPU
	inputs = {k:v.to(device) for k,v in batch.items()
				if k in tokenizer.model_input_names}
	# Extract last hidden states
	with torch.no_grad():
		last_hidden_state = model(**inputs).last_hidden_state
	# Return vector for [CLS] token
	return {"hidden_state":last_hidden_state[:, 0].cpu().numpy()}
	
```

> Since the model expects tensors as inputs, convert `input_ids` and `attention_mask` to `torch` format

```python
emotions_encoded.set_format("torch", 
							columns=["input_ids", "attention_mask", "label"])
```

> Map all the `emotions_encoded`

```python 
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
```

> Now, we have some new columns:

```python
emotions_hidden["train"].columns_names
```

```bash
['attention_mask', 'hidden_state', 'input_ids', 'label', 'text']
```

#### Creating a feature matrix

```python
X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
X_train.shape, X_valid.shape
```

```bash
((16000, 768), (2000, 768))
```

#### Training a simple classifier

```python
from sklearn.linear_model import LogiesticRegression


lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
lr_clf.score(X_valid, y_valid)
```

```bash
0.6085
```


Looking at the accuracy, it might appear that our model is just a bit better than random—but since we are dealing with an unbalanced multiclass dataset, it’s actually significantly better.

We can examine whether our model is any good by comparing it against a simple baseline.

```python
from sklearn.dummy import DummyClassifier

  

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_valid, y_valid)
```

```bash
0.352
```

Next, we go further investigating the performance of the model by looking at the confusion matrix of the classifier.

![[Pasted image 20250415155923.png]]


We can see that anger and fear are most often confused with sadness, which agrees with the observation we made when visualizing the embeddings. Also, love and surprise are frequently mistaken for joy.

### Fine-Tunning Transformers
#### Loading a pretrained model

Use `AutoModelForSequenceClassification` model instead of `AutoModel`

```python
from transformers import AutoModelForSequenceClassification


num_labels = 6
model = (AutoModelForSequenceClassification
		.from_pretrained(model_ckpt, num_lables=num_labels)
		.to(device))
```

The next step is to define the metrics that we'll use to evaluate out model's performance during fine-tunning.

#### Defining the performance metrics

`Input: EvalPrediction = (predictions, label_ids)`
`Ouput: {metric_names: values}`

```python
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
	labels = pred.label_ids
	preds = pred.predictions.argmax(-1)
	f1 = d1_score(labels, preds, average="weighted")
	acc = accuracy_score(labels, preds)
	return {"accuracy": acc, "f1-score": f1}
```

We have 2 final things to take care
- Log in Hugging Face Hub
- Define all the hyperparameters for the training run

#### Training the model

Run on a `Jupyter notebook`: 

```python
from huggingface_hub import notebook_login

notebook_login()
```

Run on a terminal

```terminal
$ huggingface-cli login
```


We use `TrainingArguments` class to define the training parameters.
The most important argument to specify is `output_dir`, which is where all the artifacts from training are stored.

Examples:
```python
from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"

training_args = TrainingArguments(output_dir=model_name,
								  num_train_epochs=2,
								  learning_rate=2e-5
								  per_device_train_batch_size=batch_suze,
								  per_device_eval_batch_size=batch_size,
								  weight_decay=0.01,
								  eval_strategy="epoch", 
								  disable_tqdm=False, 
								  logging_steps=logging_steps,
								  push_to_hub=True,
								  log_error="error")
```

Here we also set the batch size, learning rate, and number of epochs, and specify to load the best model at the end of the training run. With this final ingredient, we can instantiate and fine-tune our model with the `Trainer`:

```python
from transformers import Trainer

trainer = Trainer(model=model, args=training_args,
				  compute_metrics=compute_metrics, 
				  train_dataset=emotions_encoded["train"], 
				  eval_dataset=emotions_encoded["validation"], 
				  processing_class=processor)
trainer.train()
```

![[Pasted image 20250415162612.png]]

![[Pasted image 20250415162730.png]]


#### Error analysis
Sort the validation samples by the model loss.

```python
from torch.nn.functional import cross_entropy

  
def forward_pass_with_label(batch):
    # Place all input tensors on the same device as the model
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device),
                             reduction="none")
    # Place outputs on CPU for compatibility with other dataset columns  
    return {"loss": loss.cpu().numpy(),
            "predicted_label": pred_label.cpu().numpy()}


# Convert our dataset back to PyTorch tensors
emotions_encoded.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])
# Compute loss values
emotions_encoded["validation"] = emotions_encoded["validation"].map(
    forward_pass_with_label, batched=True, batch_size=16)


emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = (df_test["predicted_label"].apply(label_int2str))
```

Now, we can sort`emotion_encoded` by the losses:

  > [!toggle] Wrong labels
  > 
>Every process that adds labels to data can be flawed. Annotators can make mis‐ takes or disagree, while labels that are inferred from other features can be wrong. If it was easy to automatically annotate data, then we would not need a model to do it. Thus, it is normal that there are some wrongly labeled examples. With this approach, we can quickly find and correct them.

  > [!toggle] Quirks of the dataset
  > 
>Datasets in the real world are always a bit messy. When working with text, special characters or strings in the inputs can have a big impact on the model’s predictions. Inspecting the model’s weakest predictions can help identify such features, and cleaning the data or injecting similar examples can make the model more robust.

Let's have a look at the data samples with the highest losses: 
```python
df_test.sort_values("loss", ascending=False).head(10)
```

![[Pasted image 20250415173743.png]]

We can clearly see that the model predicted some of the labels incorrectly. On the other hand, it seems that there are quite a few examples with no clear class, which might be either mislabeled or require a new class altogether. In particular, joy seems to be mislabeled several times. With this information we can refine the dataset, which often can lead to as big a performance gain (or more) as having more data or larger models! 

When looking at the samples with the lowest losses, we observe that the model seems to be most confident when predicting the sadness class. Deep learning models are exceptionally good at finding and exploiting shortcuts to get to a prediction. For this reason, it is also worth investing time into looking at the examples that the model is most confident about, so that we can be confident that the model does not improperly exploit certain features of the text. So, let’s also look at the predictions with the smallest loss:

```python
df_test.sort_values("loss", ascending=True).head(10)
```

![[Pasted image 20250415173955.png]]

We now know that the joy is sometimes mislabeled and that the model is most confi‐ dent about predicting the label sadness. With this information we can make targeted improvements to our dataset, and also keep an eye on the class the model seems to be very confident about.

#### Saving and sharing the model 

```python
trainer.push_to_hub(commit_message="Training completed!")
```

#### Run inference

Since we've pushed our model to the Hub, we can now use it with the `pipeline()` function

```python
from transformers import pipeline

# Change 'transfromersbook' to your Hub username

model_id = "transformersbook/distilbert-base-cased-finetuned-emotion"
classifier = pipeline("text-classification", model=model_id)

custom_tweet = "I saw a movie today and it was really good"
preds = classifier(custom_tweetm return_all_scores=True)
```

```python
# Plot the probability of each class

preds_df = pd.DataFrame(preds[0])
plt.bar(labels, 100 * preds_df["score"], color='C0')
plt.title(f'"{custom_tweet}"')
plt.ylabel("Class probability (%)")
plt.show()
```


## Conclusion
More challenges are coming!
- Production
- Faster predictions
- More advanced and more features
- No labels? -> No fine-tuning -> What to do then?
# Reference
[[Natural-language-processing-with-transformers-revised-edition.pdf]]