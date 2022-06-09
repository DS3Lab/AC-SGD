
from transformers import GPT2TokenizerFast, DebertaV2Tokenizer

def build_tokenizer(args):
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def build_deberta_tokenizer(args):
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.tokenizer_name)
    return tokenizer
    