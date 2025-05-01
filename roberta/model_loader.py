from transformers import AutoTokenizer, AutoModelForSequenceClassification,BertTokenizer,BertForSequenceClassification,LlamaForSequenceClassification
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import torch
def get_model_and_tokenizer(model_checkpoint,task_name,device):
    num_labels = 1 if task_name == "stsb" else 3 if "mnli" in task_name  else 2
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if "llama" in model_checkpoint or "Llama" in model_checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels,device_map="auto",torch_dtype=torch.float16)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
            model.config.pad_token_id = tokenizer.pad_token_id
            print(tokenizer.pad_token,tokenizer.pad_token_id,model.config.pad_token_id)
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
            # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
            if isinstance(model, PeftModel):
                model.get_input_embeddings().weight.requires_grad = False
                model.get_output_embeddings().weight.requires_grad = False
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels).to(device)
        if model.config.pad_token_id == None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def getLoraModel(model):
    if not isinstance(model, PeftModel):
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            # target_modules,
        )
        model = get_peft_model(model, lora_config)
    return model

def loadLoraModel(model,dir=''):
    model = PeftModel.from_pretrained(model, dir)
    return model
