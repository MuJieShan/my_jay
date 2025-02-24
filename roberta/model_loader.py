from transformers import AutoTokenizer, AutoModelForSequenceClassification,BertTokenizer,BertForSequenceClassification

def get_model_and_tokenizer(model_checkpoint,task_name,device):
    num_labels = 1 if task_name == "stsb" else 3 if "mnli" in task_name  else 2
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if model.config.pad_token_id == None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer
