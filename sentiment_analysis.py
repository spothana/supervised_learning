from transformers import XLMRobertaForSequenceClassification
from datasets import load_dataset
from transformers import XLMRobertaTokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def apply_transform(row):
    text = row['sentence']
    # Use the transform_fn you retrieved in the previous cell to
    # preprocess the text
    # write your code here
    return tokenizer(text)


def predict(sequence, model, tokenizer, categories):        
    # Build a tensor of token ids out of the input sequence    
    token_ids = tokenizer(sequence, return_tensors='pt')['input_ids']

    # Set the model to the appropriate mode    
    model.eval()

    device = next(iter(model.parameters())).device
    
    # Use the model to make predictions/logits  
    pred = model(token_ids.to(device)).logits
    
    # Compute the probabilities corresponding to the logits
    # and return the top value and index    
    probabilities = torch.nn.functional.softmax(pred[0], dim=0)
    values, indices = torch.topk(probabilities, 1)
    
    return [{'label': categories[i], 'value': v.item()} for i, v in zip(indices, values)]


repo_id = "FacebookAI/xlm-roberta-base"
model = XLMRobertaForSequenceClassification.from_pretrained(repo_id, num_labels=2)
datasets = load_dataset('stanfordnlp/sst2')
row = datasets['train'][0]
text, label = row['sentence'], row['label']
tokenizer = XLMRobertaTokenizer.from_pretrained(repo_id)
#apply_transform(row)
datasets = datasets.map(apply_transform)
datasets = datasets.select_columns(['input_ids', 'label'])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataloaders = {}
dataloaders['train'] = DataLoader(dataset=datasets['train'], batch_size=16, shuffle=True, collate_fn=data_collator)
dataloaders['val'] = DataLoader(dataset=datasets['validation'], batch_size=16, shuffle=True, collate_fn=data_collator)
dl_out = next(iter(dataloaders['train']))
loss_fn = nn.CrossEntropyLoss()

lr = 1e-5
optimizer = optim.AdamW(model.parameters(), lr=lr)
writer = SummaryWriter('runs/roberta')

import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)

batch_losses = []

## Training
for i, batch in tqdm(enumerate(dataloaders['train'])):
    batch_features = batch['input_ids']
    batch_targets = batch['labels']
    batch_masks = batch['attention_mask']
    # Set the model's mode    
    model.train()
    
    # Send input_ids, labels, and attention masks to the device    
    batch_features = batch_features.to(device)
    batch_targets = batch_targets.to(device)
    batch_masks = batch_masks.to(device)
    
    # Step 1 - forward pass    
    output = model(input_ids=batch_features, 
                        attention_mask=batch_masks,
                        labels=batch_targets)
    prediction = output.logits

    # Step 2 - computing the loss    
    loss = output.loss

    # Step 3 - computing the gradients    
    loss.backward()

    batch_losses.append(loss.item())
    
    writer.add_scalars(main_tag='loss',
                       tag_scalar_dict={'training': loss.item()},
                       global_step=i)    

    # Step 4 - updating parameters and zeroing gradients
    # Tip: it takes two calls to optimizer's methods
    
    optimizer.step()
    optimizer.zero_grad()

writer.close()

## Validation   
with torch.inference_mode():
    val_losses = []

    #for i, (val_features, val_targets) in enumerate(dataloaders['val']):
    for i, val in enumerate(dataloaders['val']):
        val_features = val['input_ids']
        val_targets = val['labels']
        val_masks = val['attention_mask']
        # Set the model's mode        
        model.eval()

        # Send input_ids, labels, and attention masks to the device        
        val_features = val_features.to(device)
        val_targets = val_targets.to(device)
        val_masks = val_masks.to(device)

        # Step 1 - forward pass        
        output = model(input_ids=val_features, attention_mask=val_masks, labels=val_targets)
        predictions = output.logits

        # Step 2 - computing the loss        
        loss = output.loss        
        val_losses.append(loss.item())

categories = ['negative', 'positive']
text = "I am really liking this course"
print(predict(text, model, tokenizer, categories))

text = "This course is too complicated!"
print(predict(text, model, tokenizer, categories))