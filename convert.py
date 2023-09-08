from transformers import AutoModel
import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distill_model_path', default='../distill_model/gs80000.pkl', type=str)
    parser.add_argument('--ouput_dir', default='./bert/h256', type=str)
    args = parser.parse_args()
    distill_model = torch.load(args.distill_model_path)

    state_dict = {}
    for k,v in distill_model.items():
        if 'bert' in k:
            k = k.replace('bert.','')
            state_dict[k] = v

    bert = AutoModel.from_pretrained('./bert/h256', add_pooling_layer=False)
    bert.load_state_dict(state_dict)
    bert.save_pretrained(args.ouput_dir)

if __name__ == '__main__':
    main()