from arguments import get_args
from punc_utils import load_and_cache_examples, PUN_DICT
import torch
import random
import numpy as np
import pandas as pd
import os
from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from ltp import LTP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from model import PuncBert, PuncCRF, PuncLSTM, PuncBert_freeze
from sklearn.metrics import precision_recall_fscore_support


import logging
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

# def collate_fn(batch):
#     new_batch = { key: [] for key in batch[0].keys()}
#     for b in batch:
#         for key in new_batch:
#             new_batch[key].append(b[key]) 
#     for b in new_batch:
#         new_batch[b] = torch.tensor(new_batch[b], dtype=torch.long)
#     return new_batch

def train(args, model, train_dataset, ltp, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size 
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    args.logging_steps = eval(args.logging_steps)
    if isinstance(args.logging_steps, float):
        args.logging_steps = int(args.logging_steps * len(train_dataloader)) // args.gradient_accumulation_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    best_dev_f1 = 0.0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    set_seed(args)  # Added here for reproductibility
    model.train()
    if args.do_freeze:
        # for name, param in model.named_parameters():
        #     # print(name)
        #     if "bert.embeddings" in name:
        #         param.requires_grad = False
        #         print("已冻结",name)

        comma_state=model.bert.embeddings.word_embeddings.weight[8024]
        period_state=model.bert.embeddings.word_embeddings.weight[511]
        colon_state=model.bert.embeddings.word_embeddings.weight[8039]
        punc_state=torch.concat((comma_state,period_state,colon_state),dim=0)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):


            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if not args.do_freeze:
                inputs = {
                    "input_ids":batch[0],
                    "attention_mask":batch[1],
                    "token_type_ids":batch[2],
                    "valid_mask":batch[3],
                    "mask_labels":batch[4],
                    "mode": "train"
                }
            else:
                inputs = {
                    "input_ids":batch[0],
                    "attention_mask":batch[1],
                    "token_type_ids":batch[2],
                    "pun_state":punc_state,
                    "valid_mask":batch[3],
                    "mask_labels":batch[4],
                    "mode": "train"
                }                

            loss = model(**inputs)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    f1, _ = evaluate(args, model, ltp, tokenizer, mode = "dev")
                    if best_dev_f1 < f1:
                        best_dev_f1 = f1
                        output_dir = os.path.join(args.output_dir, "best_checkpoint")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        # logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, ltp, tokenizer, mode):
    eval_dataset = load_and_cache_examples(args, ltp, tokenizer, labels = PUN_DICT, mode = mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size 
    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    preds = []
    trues = []
    model.eval()
    if args.do_freeze:
        # for name, param in model.named_parameters():
        #     # print(name)
        #     if "bert.embeddings" in name:
        #         param.requires_grad = False
        #         print("已冻结",name)

        comma_state=model.bert.embeddings.word_embeddings.weight[8024]
        period_state=model.bert.embeddings.word_embeddings.weight[511]
        colon_state=model.bert.embeddings.word_embeddings.weight[8039]
        punc_state=torch.concat((comma_state,period_state,colon_state),dim=0)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            if not args.do_freeze:
                inputs = {
                    "input_ids":batch[0],
                    "attention_mask":batch[1],
                    "token_type_ids":batch[2],
                    "valid_mask":batch[3],
                    "mask_labels":batch[4],
                    "mode": "test"
                }
            else:
                inputs = {
                    "input_ids":batch[0],
                    "attention_mask":batch[1],
                    "token_type_ids":batch[2],
                    "pun_state":punc_state,
                    "valid_mask":batch[3],
                    "mask_labels":batch[4],
                    "mode": "test"
                }                


            pred = model(**inputs)
            label = inputs["mask_labels"][(inputs["mask_labels"] != -100)]
            preds.extend(pred.detach().cpu().tolist())
            trues.extend(label.detach().cpu().tolist())
    # O_preds=[]
    # COMMA_preds=[]
    # PERIOD_preds=[]
    # COLON_preds=[]

    # O_trues=[]
    # COMMA_trues=[]
    # PERIOD_trues=[]
    # COLON_trues=[]

    # for i in range(len(preds)):
    #     if trues[i]==0:
    #         O_trues.append(trues[i])
    #         O_preds.append(preds[i])
    #     elif trues[i]==1:
    #         COMMA_trues.append(trues[i])
    #         COMMA_preds.append(preds[i])
    #     elif trues[i]==2:
    #         PERIOD_trues.append(trues[i])
    #         PERIOD_preds.append(preds[i])
    #     elif trues[i]==3:
    #         COLON_trues.append(trues[i])
    #         COLON_preds.append(preds[i])





    precision, recall, f1, _  = precision_recall_fscore_support(trues, preds, average=None, labels = [0,1,2,3])
    overall = precision_recall_fscore_support(trues, preds, average='macro', labels = [0,1,2,3] )
    overall_f1 = overall[-2]
    eval_metric = pd.DataFrame(np.array([precision, recall, f1]),columns=[1,2,3,4],index=['Precision', 'Recall', 'F1'])
    eval_metric['OVERALL'] = overall[:3]
    print(eval_metric)

    return overall_f1, eval_metric


def main():

    args = get_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    bert = AutoModel.from_pretrained(args.model_name_or_path, add_pooling_layer=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    args.bert = bert
    args.num_labels = len(PUN_DICT)
 
    ltp = LTP('/disc1/yu/scl_output/puncs_mask/ltp')
    if torch.cuda.is_available():
        ltp.to("cuda")

    # model: BERT-Linear, BERT-BiLSTM, BERT-CRF, BERT-CNN-RNN
    if args.model_type == 'crf':
        model = PuncCRF(args)
    elif args.model_type == 'lstm':
        model = PuncLSTM(args)
    else:
        if args.do_freeze:
            model=PuncBert_freeze(args)
        else:
            model = PuncBert(args)
    model.to(args.device)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, ltp, tokenizer, labels = PUN_DICT, mode = "train")
        global_step, tr_loss = train(args, model, train_dataset, ltp, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        output_dir = os.path.join(args.output_dir, "last_checkpoint")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)

    # Evaluation
    if args.do_eval:
        checkpoint = os.path.join(args.output_dir, 'best_checkpoint')
        state_dict = torch.load(os.path.join(checkpoint, "pytorch_model.bin"))
        model.load_state_dict(state_dict)
        model.to(args.device)

        _, metric = evaluate(args, model, ltp, tokenizer, mode = "test")
        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_eval_file, "a") as writer:
            writer.write("{} \n".format(metric))

if __name__ == '__main__':
    main()