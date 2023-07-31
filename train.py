import argparse

import transformers
import datasets
import nltk
nltk.download('punkt')
import numpy as np
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from datasets import load_metric

from dataloader import load_data
from utils import clean_data, preprocess_data

def model_init(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True, stride=128)
    
    model.config.max_length = args.max_target_length
    tokenizer.model_max_length = args.max_target_length
    return model, tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter for Training T5 for summary')
    parser.add_argument('--model_checkpoint', default='paust/pko-t5-large', type=str,
                                                help='huggingface model name to train')
    parser.add_argument('--prefix', default='summarize: ', type=str,
                                                help='inference input prefix')
    parser.add_argument('--max_input_length', default=128, type=int,
                                                help='max input length for summarization')
    parser.add_argument('--max_target_length', default=32, type=int,
                                                help='max target length for summarization')
    parser.add_argument('--use_auto_find_batch_size', default=False, type=bool,
                                                help='if you want to find batch size automatically, set True')
    parser.add_argument('--train_batch_size', default=8, type=int,
                                                help='train batch size')
    parser.add_argument('--eval_batch_size', default=8, type=int,
                                                help='eval batch size')
    parser.add_argument('--num_train_epochs', default=200, type=int,
                                                help='train epoch size')
    parser.add_argument('--lr', default=4e-5, type=int,
                                                help='learning rate for training')
    parser.add_argument('--wd', default=0.01, type=int,
                                                help='weight decay for training')
    parser.add_argument('--steps', default=30000, type=int,
                                                help='evaluation, logging, saving step for training')                                            
    parser.add_argument('--model_name', default='t5-base-korean-finetuned-for-topic-extraction', type=str,
                                                help='model name for saving')
    parser.add_argument('--base_path', default='./data/', type=str,
                                                help='dataset path')
    parser.add_argument('--model_path', default='./models', type=str,
                                                help='model path for saving')
    parser.add_argument('--predict', default=True, type=bool,
                                                help='if you want to summary some example text, set True')
    args = parser.parse_args()

    # Load datset
    dataset = load_data(args.base_path)

    # Load model & tokenizer
    model, tokenizer = model_init(args)

    # Preprocessing dataset
    dataset_cleaned = dataset.filter(lambda example: (len(example['query']) >= 200) and (len(example['topic']) >= 20))
    tokenized_datasets = dataset_cleaned.map(lambda x: preprocess_data(x, tokenizer, args), batched=True)

    # Finetuning
    model_dir = f"{args.model_path}/{args.model_name}"

    if args.use_auto_find_batch_size:
        training_args = Seq2SeqTrainingArguments(
            model_dir,
            evaluation_strategy="steps", eval_steps=args.steps,
            logging_strategy="steps", logging_steps=args.steps,
            save_strategy="steps", save_steps=args.steps,
            learning_rate=args.lr,
            weight_decay=args.wd,
            auto_find_batch_size=True,
            num_train_epochs=args.num_train_epochs,
            save_total_limit=3,
            predict_with_generate=True,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
        )
    else:
        training_args = Seq2SeqTrainingArguments(
            model_dir,
            evaluation_strategy="steps", eval_steps=args.steps,
            logging_strategy="steps", logging_steps=args.steps,
            save_strategy="steps", save_steps=args.steps,
            learning_rate=args.lr,
            weight_decay=args.wd,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            save_total_limit=3,
            predict_with_generate=True,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
        )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    metric = load_metric("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                        for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                        for label in decoded_labels]
        
        # Compute ROUGE scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                                use_stemmer=True)

        # Extract ROUGE f1 scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length to metrics
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                        for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
        
    # Training
    print('Start Training...')
    
    trainer.train()

    # Saving model
    print('Saving Model...')
    trainer.save_model()

    if args.predict:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

        sample = '신발끈을 묶는 법에 관해 알려주세요'
        
        inputs = [args.prefix + sample]
        inputs = tokenizer(inputs, max_length=args.max_input_length, truncation=True, return_tensors="pt")
        output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=2, max_length=args.max_target_length)
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        result = nltk.sent_tokenize(decoded_output.strip())[0]
        print('First Example Result:')
        print(result)


        sample = '김치찌개를 끓이는 법'

        inputs = [args.prefix + sample]
        inputs = tokenizer(inputs, max_length=args.max_input_length, truncation=True, return_tensors="pt")
        output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=2, max_length=args.max_target_length)
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        result = nltk.sent_tokenize(decoded_output.strip())[0]
        print('Second Example Result:')
        print(result)
