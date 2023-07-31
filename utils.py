import nltk
import string

def clean_data(data, source_min_length=200, target_min_length=20):
    data_cleaned = data.filter(lambda example: (len(example['query']) >= source_min_length) and (len(example['topic']) >= target_min_length))
    return data_cleaned

def clean_text(text):
  sentences = nltk.sent_tokenize(text.strip())
  sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
  sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                 if len(sent) > 0 and
                                 sent[-1] in string.punctuation]
  text_cleaned = "\n".join(sentences_cleaned_no_titles)
  return text_cleaned

def preprocess_data(examples, tokenizer, args):
  texts_cleaned = [clean_text(text) for text in examples["query"]]
  inputs = [args.prefix + text for text in texts_cleaned]
  model_inputs = tokenizer(inputs, max_length=args.max_input_length, truncation=True)

  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples["topic"], max_length=args.max_target_length, 
                       truncation=True)

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

def load_data(path='./data/kullm.json'):
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_data(obj, path='./data/result_01.json'):
    import json
    with open(path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)
