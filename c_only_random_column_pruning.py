import copy
import math
import os
import time
import torch
import torchtext
from torchtext.data import get_tokenizer


from a_extract_info_from_precompression import extract_original_layers_whole_pattern
import torch.nn as nn
import numpy as np

from a_random_generate_column_whole_pattern import random_generate_column_whole_pattern
from e_heatmap import plot_heatmap
from e_load_config_file import load_config_file
from e_sparsity_ratio import sparsity_ratio
from f_transformer_model import TransformerModel


#keep batch data
def batchify(data,TEXT,device,bsz):
    data = TEXT.numericalize([data.examples[0].text])
    nbatch = data.size(0) // bsz # Divide the dataset into bsz parts.
    data = data.narrow(0, 0, nbatch * bsz) #Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.view(bsz, -1).t().contiguous()# Evenly divide the data across the bsz batches.
    sequence_length = nbatch * bsz
    return data.to(device),sequence_length


bptt = 35
#bptt is a max sentence number of one batch
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


emsize = 800
nhid = 200
nlayers = 2
nhead = 4
dropout = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset--WikiText2
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)

train_data,train_length = batchify(train_txt,TEXT,device,bsz=20)
val_data,val_length = batchify(val_txt,TEXT,device,bsz=10)
test_data,test_length = batchify(test_txt,TEXT,device,bsz=10)

ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)


#trainning
def train(bptt, model, TEXT, train_data, optimizer, criterion, scheduler,epoch,device,column_pattern_dict):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)#load data and targets
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()#update weight matrix
        for update_name, update_weight in model.named_parameters():
            if ('in_proj_weight' in update_name) or ('out_proj.weight' in update_name) \
                    or ('linear1.weight' in update_name) or ('linear2.weight' in update_name) \
                    or ('encoder.weight' in update_name) or ('decoder.weight' in update_name):
                update_weight.data = update_weight.data.mul_(column_pattern_dict[update_name].to(device))
        total_loss += loss.item()
        #print train process
        log_interval = 900
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | {:5.2f} ms/batch  | '
                    'loss {:5.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            total_loss = 0
            start_time = time.time()


#model evaluate
def evaluate(bptt, eval_model, data_source, TEXT, criterion):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    total_accuracy = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()

            _, indices = output_flat.max(1)
            accuracy = (targets==indices).sum()/float(len(targets))
            total_accuracy += len(data) * accuracy
    return total_loss / len(data_source), total_accuracy / len(data_source)


def train_prune(args,model,column_pattern_dict):
    best_accuracy= float(0)
    best_model = None

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

    print('-' * 89)
    original_loss,original_accuracy = evaluate(bptt,model,test_data,TEXT,criterion)
    print('| original model evaluate | loss {:5.2f} | accuracy {:8.4f}'.format(
        original_loss, original_accuracy))

    for update_name, update_weight in model.named_parameters():
        if ('in_proj_weight' in update_name) or ('out_proj.weight' in update_name) \
                or ('linear1.weight' in update_name) or ('linear2.weight' in update_name) \
                or ('encoder.weight' in update_name) or ('decoder.weight' in update_name):
            update_weight.data = update_weight.data.mul_(column_pattern_dict[update_name].to(device))

    print('-' * 89)
    test_loss,test_accuracy = evaluate(bptt,model,test_data,TEXT,criterion)
    print('| test before training | loss {:5.2f} | accuracy {:8.4f}'.format(test_loss, test_accuracy))

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        print('-' * 89)
        train(bptt, model, TEXT, train_data, optimizer, criterion, scheduler, epoch, device, column_pattern_dict)
        test_loss,test_accuracy = evaluate(bptt,model,val_data,TEXT,criterion)
        print('| end of epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | accuracy {:8.4f}'.format(epoch, (time.time() - epoch_start_time),test_loss,test_accuracy))
        # save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = model

        scheduler.step()
    print('-' * 89)
    reward_loss, reward_accuracy = evaluate(bptt, model, test_data, TEXT, criterion)
    print('| end of training | loss {:5.2f} | accuracy {:8.4f}'.format(reward_loss, reward_accuracy))
    print('-' * 89)
    torch.save(best_model.state_dict(), args.save_file)
    print('random column pruning model saved!')

    return reward_accuracy

def main(args):
    prune_ratios = load_config_file(args.config_file)
    print("prune_ratios:",prune_ratios)

    if args.random:
        print('#' * 89)
        print('A.only column pruning from 50epochs model')
        print('B.random generate column pattern')
        print('C.fine-tune {} epochs'.format(args.epochs))
        print('#' * 89)
        model.load_state_dict(torch.load('./model/transformer_model_lr_3.0_50.pt'))
        column_whole_pattern_dict = random_generate_column_whole_pattern(model,prune_ratios,args.block_size)
        train_prune(args,model,column_whole_pattern_dict)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Random Column pruning')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epoch to run')
    parser.add_argument('--lr',default=0.23,type=float,
                        help='learning rate')
    parser.add_argument('--block-size',default=100,type=int,metavar='N',
                        help='block size')
    parser.add_argument('--config-file',default='./config_file/prune_ratio_v1.yaml',
                        help='config file of prune ratios for every layers')
    parser.add_argument('--random',action='store_true',help='generate column pattern random or not')
    parser.add_argument('--save-file',default='./model/random_column_pruning_average.pt',
                        help='model save location')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

#CUDA_VISIBLE_DEVICES=3 nohup python -u c_only_random_column_pruning.py --epochs 10 --random > only_random_column_pruning_average