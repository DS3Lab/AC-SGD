from comm.comm_utils import *
import datasets
# import wandb

def get_metric(args):
    metrics = []
    if args.task_name == 'cola':
        metric = datasets.load_metric('./metrics/matthews_correlation')
        metrics.append(metric)
        metric = datasets.load_metric('./metrics/accuracy')
        metrics.append(metric)
    if args.task_name in {'qnli', 'qqp', 'mrpc', 'mnli', 'sst2'}:
        metric = datasets.load_metric('./metrics/accuracy')
        metrics.append(metric)
        metric = datasets.load_metric('./metrics/f1')
        metrics.append(metric)
    if args.task_name in {'wikitext', 'wiki103', 'arxiv21'}:
        metric = datasets.load_metric('./metrics/perplexity_custom')
        metrics.append(metric)
    return metrics

def distributed_test_foo_iter(args, pipeline, device, test_data_loader):
    pipeline.model.eval()
    if get_pipeline_parallel_rank() == 0:
        for i, data in enumerate(test_data_loader):
            input_ids = data['text'].to(device)
            current_iter_time = pipeline.infer_iter(input_ids, None, None)
    elif get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
        metrics = get_metric(args)
        for i, data in enumerate(test_data_loader):
            input_ids = data['text'].to(device)
            labels = data['label'].to(device)
            pipeline.infer_iter(input_ids, labels, None, metrics=metrics)
        
        print({
            metric.name: metric.compute() for metric in metrics
        })
#         wandb.log(
#             {metric.name: metric.compute() for metric in metrics}, 
#             step=pipeline.global_step,
#         )
    else:
        for i, data in enumerate(test_data_loader):
            pipeline.infer_iter(None, None, None)
            
            
            
def _lm_pred_func(x, y):
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    logits = x[:, :-1, :].contiguous()
    labels = y[:, 1:].contiguous()
    loss = loss_fct(logits.transpose(-1, -2), labels).mean(1).detach().cpu()
    return loss
            
def distributed_test_lm_iter(args, pipeline, device, test_data_loader):
    pipeline.model.eval()
    if get_pipeline_parallel_rank() == 0:
        for i, data in enumerate(test_data_loader):
            input_ids = data['text'].to(device)
            current_iter_time = pipeline.infer_iter(input_ids, None, None)
    elif get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
        metrics = get_metric(args)
        for i, data in enumerate(test_data_loader):
            labels = data['text'].to(device)
            pipeline.infer_iter(None, labels, None, 
                                metrics=metrics, pred_func=_lm_pred_func)
        
        print({
            metric.name: metric.compute() for metric in metrics
        })
#         wandb.log(
#             {metric.name: metric.compute() for metric in metrics}, 
#             step=pipeline.global_step,
#         )
    else:
        for i, data in enumerate(test_data_loader):
            pipeline.infer_iter(None, None, None)
            
            
            
def distributed_test_bert_iter(args, pipeline, device, test_data_loader):
    pipeline.model.eval() # Flag .training to True to enable Dropout
    if get_pipeline_parallel_rank() == 0:
        for i, data in enumerate(test_data_loader):
            inputs_ids = data['text'].to(device)
            aux_inputs = {
                'token_type_ids': data['token_type_ids'].to(device),
                'attention_mask': data['attention_mask'].to(device),
            }
            current_iter_time = pipeline.infer_iter(inputs_ids, aux_input_data=aux_inputs)
    elif get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
        metrics = get_metric(args)
        for i, data in enumerate(test_data_loader):
            aux_inputs = {
                'attention_mask': data['attention_mask'].to(device),
            }
            input_ids = data['text'].to(device)
            labels = data['label'].to(device)
            pipeline.infer_iter(None, labels, aux_input_data=aux_inputs, metrics=metrics)
        
        print({
            metric.name: metric.compute() for metric in metrics
        })
#         wandb.log(
#             {metric.name: metric.compute() for metric in metrics}, 
#             step=pipeline.global_step,
#         )
    else:
        for i, data in enumerate(test_data_loader):
            aux_inputs = {
                'attention_mask': data['attention_mask'].to(device),
            }
            pipeline.infer_iter(aux_input_data=aux_inputs)