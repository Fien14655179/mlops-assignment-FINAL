import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader
import random 
import typing

# we zetten alle random seeds vast zodat de uitkomst elke keer hetzelfde is
# dit maakt het makkelijker om resultaten te vergelijken en te debuggen
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)

# we kiezen hoeveel punten we willen genereren per groep
num_samples = 1000

# we maken twee groepen punten in 2D
# groep A zit rond (0,0) en groep B zit rond (3,3)
# 0.75 is hier de spreiding: hoe groter, hoe meer de wolk uit elkaar valt
A = np.random.normal(0, 0.75, size=(num_samples, 2))
B = np.random.normal(3, 0.75, size=(num_samples, 2))

# we maken labels voor de punten
# label 0 hoort bij groep A en label 1 hoort bij groep B
# we maken in totaal 2000 labels (1000 voor A en 1000 voor B)
y = [0 if i < num_samples else 1 for i in range(num_samples * 2)]

# we plakken alle punten onder elkaar zodat we één grote dataset hebben
# input_data krijgt vorm (2000, 2)
input_data = np.concatenate([A, B], axis=0)

# we plotten de data om te zien of de verdeling klopt
# c=y betekent dat punten een kleur krijgen op basis van hun label
plt.figure(figsize=(10, 7))
plt.scatter(input_data[:, 0], input_data[:, 1], c=y, alpha=0.5)
plt.title("Generated Data Distribution")
plt.grid()
plt.savefig("data_distribution.png")
plt.close() 

class Model(nn.Module):
    def __init__(self, input_dim: int=2, hidden_dim: int=5, output: int=1):
        super().__init__()
        
        # eerste laag: van input_dim naar hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # tweede laag: van hidden_dim naar output (1 getal: kans op class 1)
        self.fc2 = nn.Linear(hidden_dim, output)

        # activatie functies
        # ReLU maakt negatieve waarden 0 en laat positieve waarden door
        self.relu = nn.ReLU()
        # sigmoid zet een waarde om naar een kans tussen 0 en 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward betekent: hoe gaat data door het netwerk heen
        output = self.fc1(x) 
        output = self.relu(output)
        output = self.fc2(output)
        sigmoid = self.sigmoid(output)
        # we geven uiteindelijk de kans terug (tussen 0 en 1)
        return sigmoid 

class ExampleData(Dataset):
    def __init__(self, datapoints: np.array, labels: np.array):
        # we zetten de numpy arrays om naar torch tensors
        # zo kan PyTorch ermee trainen
        self.datapoints = torch.tensor(datapoints)
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        # hiermee weet de DataLoader hoeveel items er zijn
        return len(self.datapoints)

    def __getitem__(self, index: int):
        # hiermee kan de DataLoader één datapunt opvragen
        # we zorgen dat het float is zodat het kan rekenen in het model
        x = self.datapoints[index].float()
        y = self.labels[index].float()
        return x, y

# hieronder maken we nóg een model, maar nu met nn.Sequential
# let op: dit overschrijft het Model hierboven, want we gebruiken dezelfde naam 'model'
# dit netwerk is veel groter: input (2) -> 1024 neuronen -> output (1)
# dropout p=0.5 betekent dat tijdens training de helft van de neuronen random wordt uitgezet
# dat helpt tegen overfitting
model = nn.Sequential(nn.Linear(2, 1024), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1024, 1), nn.Sigmoid()) 
print(model)

# we maken een Dataset object met alle punten en labels
train_data = ExampleData(input_data, y)

# DataLoader zorgt dat we batches krijgen in plaats van alles tegelijk
# batch_size=64 betekent dat we steeds 64 punten tegelijk gebruiken
# shuffle=True zorgt dat de volgorde elke epoch opnieuw random is
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# optimizer bepaalt hoe de weights veranderen
# Adam is een slimme optimizer die vaak sneller convergeert dan SGD
optimizer = Adam(model.parameters(), lr=0.001)

# BCELoss is Binary Cross Entropy loss
# dit past bij sigmoid output en labels 0/1
criterion = nn.BCELoss()

# hier bewaren we de loss per training stap
loss_list = []

# we trainen 5 epochs
for epoch in range(5):
    # elke epoch lopen we door alle batches heen
    for x, y in train_loader:
        # eerst gradients resetten
        optimizer.zero_grad()

        # forward pass: voorspelling maken
        # squeeze(dim=1) haalt de extra dimensie weg zodat het goed matcht met y
        output = model(x).squeeze(dim=1)

        # loss berekenen tussen output (kans) en echte y (0/1)
        loss = criterion(output, y)

        # backward pass: gradients berekenen
        loss.backward()

        # update stap: weights aanpassen
        optimizer.step()

        # loss opslaan zodat we later de curve kunnen plotten
        loss_list.append(loss.item())


# plot de loss over alle iteraties (dus niet per epoch gemiddeld, maar echt per batch stap)
plt.figure(figsize=(10, 5))
plt.plot(loss_list, label="Raw Loss", alpha=0.3)
plt.title("Training Loss Over Time")
plt.xlabel("Iteration")
plt.ylabel("BCE Loss")
plt.legend()
plt.savefig("training_loss_dataloader_dropout_0.5.png")