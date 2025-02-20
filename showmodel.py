from strategies import pruner
from utils import *
from dataloader import get_dataloader
from transformers import get_cosine_schedule_with_warmup
import copy
import torch
def main():
    config = init_config()
    log=init_log()
    log.info(config)
    seed_torch(config.seed)

    # Config Settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_checkpoint = "roberta-base"
    # model_checkpoint = "bert-base-uncased"
    task = config.dataset

    # Load Pre-trained Model
    print(f"\nLoading pre-trained BERT model \"{model_checkpoint}\"")
    model=load_model(model_checkpoint,task,device)
    # print(torch.sum(model.bert.encoder.layer[0].attention.self.query.weight.data).item())
    # print(torch.sum(model.linear.weight.data).item())##两者模型一致

    train_dataloader,_=get_dataloader(task, model_checkpoint, shuffle=False,batch_size=32)
    iterator = iter(train_dataloader)
    inputs = prepare_inputs(next(iterator), device)
    print(f'input_ids:{torch.sum(inputs["input_ids"])}')#输入一致
    model.eval()
    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, Data type: {param.dtype}")#模型参数精度一致
    #     break
    step_loss, logit, step_metric, step_metric_1, _ = compute_loss(model, inputs)
    print(f"Step Loss: {step_loss}{step_loss.dtype}")
    print(f"logit: {logit}{logit.dtype}")
    # with torch.no_grad():
    #     for name, module in model.named_modules():
    #         if isinstance(module, torch.nn.Linear):
    #             print(name)

if __name__ == "__main__":
    main()


