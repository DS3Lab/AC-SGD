
import os
import torch
from datasets import load_dataset, load_from_disk


def get_qnli_data_loader(args, tokenizer, data_split='train', num_workers=0):
    
    
    def _encode(examples):
        return tokenizer(examples['question'], examples['sentence'], 
                         truncation=True, padding='max_length', max_length=args.seq_length)
    
    if os.path.isdir('./data/glue_qnli'):
        train_set = load_from_disk('./data/glue_qnli')[data_split]
    else:
        train_set = load_dataset('glue', 'qnli', split=data_split)
    train_set = train_set.map(_encode, batched=True)
    train_set = train_set.map(lambda examples: {'text': examples['input_ids']}, batched=True)
    if 'token_type_ids' in train_set.features:
        train_set.set_format(
            type='torch', columns=[
                'text', 'input_ids', 'token_type_ids', 'attention_mask', 'label', 'idx',
            ])
    else:
        train_set.set_format(
            type='torch', columns=[
                'text', 'input_ids', 'attention_mask', 'label', 'idx',
            ])
    
    
    if data_split == 'train':
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        train_sampler = torch.utils.data.RandomSampler(train_set, generator=generator)
        train_data_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=args.batch_size,
                                                        sampler=train_sampler,
                                                        shuffle=False,
                                                        num_workers=num_workers,
                                                        drop_last=True,
                                                        pin_memory=True,
                                                        collate_fn=None)
    else:
        # test or valid data loader
        # TODO: let drop_last be False
        train_data_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=args.batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers,
                                                        drop_last=True,
                                                        pin_memory=True,
                                                        collate_fn=None)
    return train_data_loader