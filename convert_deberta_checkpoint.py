import os
import argparse
import torch
from transformers import DebertaV2Model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--model-name', type=str, default='deberta-v3-base', 
                        help='model-name')
    parser.add_argument('--save-dir', type=str, default='checkpoints', 
                        help='model-name')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, args.model_name)
    
    model = DebertaV2Model.from_pretrained(args.model_name)
#     model.save_pretrained(save_path)

    torch.save(model.embeddings.state_dict(), os.path.join(save_path, 'pytorch_embs.pt'))

    for i in range(len(model.encoder.layer)):
        torch.save(model.encoder.layer[i].state_dict(), os.path.join(save_path, f'pytorch_{i}.pt'))
        
    if hasattr(model.encoder, 'rel_embeddings'):
        torch.save(model.encoder.rel_embeddings.state_dict(), os.path.join(save_path, 'pytorch_rel_embs.pt'))
    if hasattr(model.encoder, 'LayerNorm'):
        torch.save(model.encoder.LayerNorm.state_dict(), os.path.join(save_path, 'pytorch_ln.pt'))
    if hasattr(model.encoder, 'conv') and model.encoder.conv is not None:
        torch.save(model.encoder.conv.state_dict(), os.path.join(save_path, 'pytorch_conv.pt'))