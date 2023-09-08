import torch
import time
import argparse
import os
import random
from transformers import AutoTokenizer, AutoModel
from model import PuncCRF, PuncBert, PuncLSTM
from ltp import LTP
import json
def process_example(text, ltp, model, tokenizer, max_seq_length):
    start_time = time.time()
    # print(text)
    tokenize_words = ltp.pipeline(text, tasks=["cws"]).cws
    # tokens = []
    tokens_mask = []
    for token in tokenize_words:
        tokens_mask.append(token)
        tokens_mask.append('[MASK]')
    tokens = tokenizer.tokenize(''.join(tokens_mask))
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    mask_positions = [idx for idx,t in enumerate(tokens) if t == '[MASK]' ]
    valid_mask = [0] * len(tokens)
    for i in mask_positions:
        valid_mask[i] = 1
    # tokens = tokens[1:-1]
    # inputs = tokenizer(''.join(tokens) , return_tensors="pt")
    inputs = {}
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_type_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)
    inputs['input_ids'] = torch.tensor(input_ids).unsqueeze(0)
    inputs['token_type_ids'] = torch.tensor(token_type_ids).unsqueeze(0)
    inputs['attention_mask'] = torch.tensor(attention_mask).unsqueeze(0)
    inputs['valid_mask'] = torch.tensor(valid_mask).unsqueeze(0)

    # assert len(valid_mask) == len(inputs['input_ids'])
    outputs = model(**inputs)

    pred = outputs.tolist()
    mapping = {1:'，',2:'。'}
    # print(pred)
    tokens = tokens[1:-1]
    print(''.join(tokens))
    new_tokens = []
    idx = 0
    for token in tokens_mask:
        if token != '[MASK]':
            new_tokens.append(token)
        else:
            if pred[idx] != 0:
                new_tokens.append(mapping[pred[idx]])
            idx += 1
            if idx >= len(pred):
                break
    end_time = time.time()
    print(end_time-start_time)
    return ''.join(new_tokens)

def print_true(tokens, tags):
    mapping = {'COMMA':'，','PERIOD':'。'}
    new_tokens = []
    for token,tag in zip(tokens, tags):
        new_tokens.append(token)
        if tag in ['COMMA','PERIOD']:
            new_tokens.append(mapping[tag])
    return ''.join(new_tokens)

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--model_path", default='./output/best_checkpoint', type=str)
    parser.add_argument("--ltp_path", default='../ltp/', type=str)
    parser.add_argument("--model", default='bert', type=str, help='bert, crf, lstm')
    # parser.add_argument("--input_text", default="体格检查舌质暗红舌下络脉瘀紫苔薄黄脉弦细", type=str)
    args = parser.parse_args()

    ltp = LTP(args.ltp_path)
    # if torch.cuda.is_available():
    #     ltp = ltp.to("cuda")
    bert = AutoModel.from_pretrained(args.model_path, add_pooling_layer=False)
    args.bert = bert
    args.num_labels = 3
    args.dropout_prob = 0.1

    if args.model == 'bert':
        model = PuncBert(args)
    elif args.model == 'lstm':
        model = PuncLSTM(args)
    else:
        model = PuncCRF(args)
    state_dict = torch.load(os.path.join(args.model_path, "pytorch_model.bin"))
    model.load_state_dict(state_dict)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # result = process_example(args.input_text, ltp, model, tokenizer)
    # print(result)
    test_path = './data/test.json'
    with open(test_path,'r', encoding='utf-8') as f:
        test_data = json.load(f)
        random.shuffle(test_data)
        test_data = test_data[:300]
        for sample in test_data:
            tokens = sample['tokens']
            tags = sample['tags']
            text = ''.join(tokens)
            print('-'*50)
            print('input:', text)
            pred = process_example(text, ltp, model, tokenizer, 512)
            print('pred:',pred)
            true = print_true(tokens, tags)
            print('true:',true)
            

if __name__ == '__main__':
    main()