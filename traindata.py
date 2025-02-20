from strategies import pruner
from utils import *
from dataloader import get_dataloader, get_dataloader2
from transformers import get_cosine_schedule_with_warmup
import copy
import time
def main():
    start_time = time.time()
    config = init_config()
    log=init_log()
    log.info(config)
    seed_torch(config.seed)

    # Config Settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_checkpoint = "bert-base-uncased"
    task = config.dataset
    batch_size = config.batchsize
    steps = config.epoch

    lr = config.learning_rate
    # Load DataLoader
    print(f"\nLoading data...")
    train_epoch_iterator, eval_epoch_iterator ,trainset= get_dataloader2(task, model_checkpoint, shuffle=True,batch_size=batch_size)

    # Load Pre-trained Model
    print(f"\nLoading pre-trained BERT model \"{model_checkpoint}\"")
    model=load_model(model_checkpoint,task,device)
    # Define optimizer and lr_scheduler
    optimizer = create_optimizer(model, learning_rate=lr,config=config,batch_num=len(train_epoch_iterator))
    # optimizer1 = create_optimizer(model1, learning_rate=lr,config=config,batch_num=len(train_epoch_iterator))
    print('train data len:', len(train_epoch_iterator))
    # train_eval_loop(config, model, train_epoch_iterator, eval_epoch_iterator, optimizer, device, log)
    train_ft_loop2(config, model, train_epoch_iterator, eval_epoch_iterator, optimizer, device, log,trainset)
    end_time = time.time()
    total_time = end_time - start_time
    s = f'Total training time: {total_time}'
    log.info(s)

if __name__ == "__main__":
    main()


