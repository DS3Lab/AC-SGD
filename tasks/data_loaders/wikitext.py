import os
import re
import torch
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from comm.comm_utils import *


def wikitext_detokenize(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string

    

    
def get_wikitext_train_data_loader(args, tokenizer, num_workers=0):
    
    if os.path.isdir("./data/wikitext/train"):
        data = load_from_disk("./data/wikitext/train")
    else:
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    encodings = tokenizer("\n\n".join(
        [wikitext_detokenize(t) for t in data["text"]]
    ), return_tensors="pt")
    
    input_ids_list = []
    stride = args.seq_length
    for i in range(0, encodings.input_ids.size(1)-stride, stride):
        begin_loc = i
        end_loc = min(i+stride, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        input_ids_list.append(input_ids)
    input_ids = torch.cat(input_ids_list, 0)
    
    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        dp_rank = get_data_parallel_rank()
        n_samples = len(input_ids)
        n_samples_per_rank = n_samples // args.data_group_size
        i_begin, i_end = dp_rank * n_samples_per_rank, (dp_rank+1) * n_samples_per_rank
        input_ids = input_ids[i_begin: i_end]
    else:
        dp_rank = 0
    
    train_set = Dataset.from_dict({
        'input_ids': input_ids,
        'attention_mask': torch.ones_like(input_ids),
        'idx': list(range(len(input_ids))),
    })
    
    train_set = train_set.map(lambda examples: {'text': examples['input_ids']}, batched=True)
    train_set.set_format(
        type='torch', columns=[
            'text', 'input_ids', 'attention_mask', 'idx',
        ])
    
    generator = torch.Generator()
    generator.manual_seed(args.seed + dp_rank)
    train_sampler = torch.utils.data.RandomSampler(train_set, generator=generator)
    train_data_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=args.batch_size,
                                                    sampler=train_sampler,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader
    
    
def get_wikitext_test_data_loader(args, tokenizer, num_workers=0):
    
    if os.path.isdir("./data/wikitext/test"):
        data = load_from_disk("./data/wikitext/test")
    else:
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        
    encodings = tokenizer("\n\n".join(
        [wikitext_detokenize(t) for t in data["text"]]
    ), return_tensors="pt")
    
    input_ids_list = []
#     window = args.seq_length # TODO: a smaller value
#     for i in range(window, encodings.input_ids.size(1)):
#         begin_loc = max(i - window, 0)
#         end_loc = min(i, encodings.input_ids.size(1))
#         input_ids = encodings.input_ids[:, begin_loc:end_loc]
#         input_ids_list.append(input_ids)
#     input_ids = torch.cat(input_ids_list, 0)
    stride = args.seq_length
    # TODO: last stride is dropped
    for i in range(0, encodings.input_ids.size(1)-stride, stride):
        begin_loc = i
        end_loc = min(i+stride, encodings.input_ids.size(1))
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        input_ids_list.append(input_ids)
    input_ids = torch.cat(input_ids_list, 0)
    
    train_set = Dataset.from_dict({
        'input_ids': input_ids,
        'attention_mask': torch.ones_like(input_ids),
        'idx': list(range(len(input_ids))),
    })
    
    train_set = train_set.map(lambda examples: {'text': examples['input_ids']}, batched=True)
    train_set.set_format(
        type='torch', columns=[
            'text', 'input_ids', 'attention_mask', 'idx',
        ])
    
    # TODO: let drop_last be False
    train_data_loader = torch.utils.data.DataLoader(train_set,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    collate_fn=None)
        
    return train_data_loader
