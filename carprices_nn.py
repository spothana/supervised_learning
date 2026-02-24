import os
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset, Split, DatasetDict

def filter_for_data(filename):
    return ("unclean" not in filename) and ("focus" not in filename) and ("cclass" not in filename) and filename.endswith(".csv")

def gen_encoder_dict(dataset, col):
    values = sorted(dataset.unique(col))
    values += ['UNKNOWN']
    return dict(zip(values, range(len(values))))

colnames = ['model', 'year', 'price', 'transmission', 'mileage', 'fuel_type', 'road_tax', 'mpg', 'engine_size']#, 'manufacturer']

folder = './car_prices'
data_files = sorted([os.path.join(folder, fname) for fname in os.listdir(folder) if filter_for_data(fname)])

dataset = load_dataset("csv", data_files=data_files, 
                       sep=',', 
                       skiprows=1, 
                       column_names=colnames,
                       split=Split.ALL)

train_test = dataset.train_test_split(train_size=0.8)
val_test = train_test['test'].train_test_split(train_size=0.5)

datasets = DatasetDict({'train': train_test['train'],
                       'val': val_test['train'],
                       'test': val_test['test']})

cont_attr = ['year', 'mileage', 'road_tax', 'mpg', 'engine_size']
cat_attr = ['model', 'transmission', 'fuel_type']

dropdown_encoders = {col: gen_encoder_dict(datasets['train'], col) for col in cat_attr}

def preproc(row):
    colnames = ['model', 'year', 'price', 'transmission', 'mileage', 'fuel_type', 'road_tax', 'mpg', 'engine_size']#, 'manufacturer']
    
    cat_attr = ['model', 'transmission', 'fuel_type']
    cont_attr = ['year', 'mileage', 'road_tax', 'mpg', 'engine_size']
    target = 'price'
    
    cont_X = [float(row[name]) for name in cont_attr]
    cat_X = [dropdown_encoders[name].get(row[name], dropdown_encoders[name]['UNKNOWN']) for name in cat_attr]
            
    return {'label': np.array([float(row[target])], dtype=np.float32),
            'cont_X': np.array(cont_X, dtype=np.float32), 
            'cat_X': np.array(cat_X, dtype=int)}

datasets = datasets.map(preproc)
datasets = datasets.select_columns(['label', 'cont_X', 'cat_X'])
datasets = datasets.with_format('torch')

dataloaders = {}
dataloaders['train'] = DataLoader(dataset=datasets['train'], batch_size=128, drop_last=True, shuffle=True)
dataloaders['val'] = DataLoader(dataset=datasets['val'], batch_size=128)
dataloaders['test'] = DataLoader(dataset=datasets['test'], batch_size=128)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, n_cont, cat_list, emb_dim):
        super().__init__()
        
        # Embedding layers
        embedding_layers = []
        # Creates one embedding layer for each categorical feature

        # write your code here
        for categories in cat_list:
            embedding_layers.append(nn.Embedding(len(categories), emb_dim))        

        self.emb_layers = nn.ModuleList(embedding_layers)

        # Total number of embedding dimensions
        self.n_emb = len(cat_list) * emb_dim
        self.n_cont = n_cont
        # Batch Normalization layer for continuous features
        self.bn_input = nn.BatchNorm1d(n_cont)

        # Linear Layer(s)
        lin_layers = []
        # The input layers takes as many inputs as the number of continuous features
        # plus the total number of concatenated embeddings
        # The number of outputs is your own choice
        # Optionally, add more hidden layers, don't forget to match the dimensions if you do
        # write your code here
        lin_layers.append(nn.Linear(self.n_emb + self.n_cont, 100))
        self.lin_layers = nn.ModuleList(lin_layers)

        # Batch Normalization Layer(s)
        bn_layers = []
        # Creates batch normalization layers for each linear hidden layer

        # write your code here
        bn_layers.append(nn.BatchNorm1d(100))
        self.bn_layers = nn.ModuleList(bn_layers)
        
        # The output layer must have as many inputs as there were outputs in the last hidden layer
        # write your code here
        self.output_layer = nn.Linear(self.lin_layers[-1].out_features, 1)

        # Layer initialization
        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data, nonlinearity='relu')
        nn.init.kaiming_normal_(self.output_layer.weight.data, nonlinearity='relu')

    def forward(self, inputs):
        # The inputs are the features as returned in the first element of a tuple
        # coming from the dataset/dataloader
        # Make sure you split it into continuous and categorical attributes according
        # to your dataset implementation of __getitem__
        # write your code here
        cont_data, cat_data = inputs['cont_X'], inputs['cat_X']
        
        # Retrieve embeddings for each categorical attribute and concatenate them
        embeddings = []
        # write your code here
        for i, layer in enumerate(self.emb_layers):
            embeddings.append(layer(cat_data[:, i]))
        embeddings = torch.cat(embeddings, 1)
        
        # Normalizes continuous features using Batch Normalization layer
        normalized_cont_data = self.bn_input(cont_data)
        
        # Concatenate all features together, normalized continuous and embeddings
        # write your code here
        x = torch.cat([normalized_cont_data, embeddings], 1)
        
        # Run the inputs through each layer and applies an activation function and batch norm to each output
        for layer, bn_layer in zip(self.lin_layers, self.bn_layers):
            # write your code here
            x = layer(x)
            x = F.relu(x)
            x = bn_layer(x)
            
        # Run the output of the last linear layer through the output layer
        # write your code here
        x = self.output_layer(x)
        
        # Return the prediction
        # write your code here
        return x
    

n_cont = len(cont_attr)
cat_list = [np.array(list(dropdown_encoders[name].values())) for name in cat_attr]

n_cont, cat_list


torch.manual_seed(42)

# write your code here
emb_dim = 5
model = FFN(n_cont, cat_list, emb_dim=emb_dim)

# write your code here
loss_fn = nn.MSELoss()

# Suggested learning rate
lr = 3e-3

# write your code here
optimizer = optim.Adam(model.parameters(), lr=lr)

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_epochs = 20

losses = torch.empty(n_epochs)
val_losses = torch.empty(n_epochs)

best_loss = torch.inf
best_epoch = -1
patience = 3

model.to(device)

progress_bar = tqdm(range(n_epochs))

for epoch in progress_bar:
    batch_losses = []
    
    ## Training
    for i, batch in enumerate(dataloaders['train']):
        # Set the model to training mode
        # write your code here
        model.train()
                
        # Send batch features and targets to the device
        # write your code here
        batch['cont_X'] = batch['cont_X'].to(device)
        batch['cat_X'] = batch['cat_X'].to(device)
        batch['label'] = batch['label'].to(device)
        
        # Step 1 - forward pass
        # write your code here
        predictions = model(batch)

        # Step 2 - computing the loss
        # write your code here
        loss = loss_fn(predictions, batch['label'])

        # Step 3 - computing the gradients
        # Tip: it requires a single method call to backpropagate gradients
        # write your code here
        loss.backward()

        batch_losses.append(loss.item())

        # Step 4 - updating parameters and zeroing gradients
        # Tip: it takes two calls to optimizer's methods
        # write your code here
        optimizer.step()
        optimizer.zero_grad()
        
    losses[epoch] = torch.tensor(batch_losses).mean()

    ## Validation   
    with torch.inference_mode():
        batch_losses = []

        for i, val_batch in enumerate(dataloaders['val']):
            # Set the model to evaluation mode
            # write your code here
            model.eval()

            # Send batch features and targets to the device
            # write your code here
            val_batch['cont_X'] = val_batch['cont_X'].to(device)
            val_batch['cat_X'] = val_batch['cat_X'].to(device)
            val_batch['label'] = val_batch['label'].to(device)

            # Step 1 - forward pass
            # write your code here
            predictions = model(val_batch)

            # Step 2 - computing the loss
            # write your code here
            loss = loss_fn(predictions, val_batch['label'])

            batch_losses.append(loss.item())

        val_losses[epoch] = torch.tensor(batch_losses).mean()
        
        if val_losses[epoch] < best_loss:
            best_loss = val_losses[epoch]
            best_epoch = epoch
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       'best_model.pth')
        elif (epoch - best_epoch) > patience:
            print(f"Early stopping at epoch #{epoch}")
            break


import matplotlib.pyplot as plt

plt.plot(losses[:epoch], label='Training')
plt.plot(val_losses[:epoch], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.savefig('carprices_nn_losses.png')



split = 'val'
y_hat = []
y_true = []
for batch in dataloaders[split]:
    model.eval()
    batch['cont_X'] = batch['cont_X'].to(device)
    batch['cat_X'] = batch['cat_X'].to(device)
    batch['label'] = batch['label'].to(device)
    y_hat.extend(model(batch).tolist())
    y_true.extend(batch['label'].tolist())
    
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(y_true, y_hat, alpha=0.25)
ax.plot([0, 80000], [0, 80000], linestyle='--', c='k', linewidth=1)
ax.set_xlabel('Actual')
ax.set_xlim([0, 80000])
ax.set_ylabel('Predicted')
ax.set_ylim([0, 80000])
ax.set_title('Price')    

figure = fig.get_figure()
figure.savefig('carprices_nn_predictions.png')