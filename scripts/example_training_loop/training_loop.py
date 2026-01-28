import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from ml_core.data import get_dataloaders
from ml_core.models import MLP

# hier zetten we alles wat we nodig hebben in één configuratie-dict
# zo kunnen we later makkelijk dingen aanpassen zonder overal in de code te zoeken
config = {
    # instellingen voor de data: waar het staat en hoe we het in batches inladen
    "data": {"data_path": "../data/pcam/", "batch_size": 32, "num_workers": 2},
    # instellingen voor het model: hoe groot de input is en hoe het netwerk er van binnen uitziet
    "model": {"input_shape": [3, 96, 96], "hidden_units": [64, 32], "num_classes": 2},
}

# we kiezen of we op de GPU (cuda) trainen of op de CPU
# als cuda beschikbaar is dan gaat het meestal veel sneller
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

# we maken nu de dataloaders aan
# train_loader levert batches voor training
# val_loader levert batches voor validatie (controle of het model generaliseert)
train_loader, val_loader = get_dataloaders(config)

# we bouwen het model op basis van de config en zetten het op de juiste device
# .to(device) zorgt dat zowel de berekeningen als de gewichten op gpu/cpu staan
model = MLP(**config["model"]).to(device)

# optimizer bepaalt hoe we de gewichten gaan updaten
# SGD is een simpele en klassieke keuze
optimizer = optim.SGD(model.parameters(), lr=0.001)

# dit is de loss functie voor classificatie met meerdere classes
# cross entropy vergelijkt de modelvoorspellingen met de echte labels
criterion = nn.CrossEntropyLoss()

# hier gaan we per epoch de gemiddelde loss opslaan om later te plotten
train_losses = []
val_losses = []

# we trainen hier 3 epochs
# een epoch betekent: één keer door de volledige training set heen
for epoch in range(3):
    # zet het model in train-modus
    # dit is belangrijk omdat sommige lagen (zoals dropout) zich anders gedragen in training
    model.train()
    epoch_train_loss = 0

    # loop over alle batches in de training loader
    for i, (images, labels) in enumerate(train_loader):
        # verplaats de data ook naar dezelfde device als het model
        images, labels = images.to(device), labels.to(device)

        # stap 1: reset de gradients van de vorige stap
        # anders zouden de gradients zich opstapelen
        optimizer.zero_grad()

        # stap 2: forward pass: we halen voorspellingen uit het model
        outputs = model(images)

        # stap 3: bereken de loss door outputs te vergelijken met labels
        loss = criterion(outputs, labels)

        # stap 4: backward pass: bereken hoe elke weight moet veranderen
        loss.backward()

        # stap 5: optimizer update: pas de weights echt aan
        optimizer.step()

        # we tellen de loss op om later een gemiddelde te kunnen nemen
        epoch_train_loss += loss.item()

        # af en toe printen we een status update zodat je ziet dat het loopt
        # i % 100 == 0 betekent: bij stap 0, 100, 200, ...
        if i % 100 == 0: 
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

    # gemiddelde training loss voor deze epoch
    # door te delen door het aantal batches krijg je een schaalbaar getal
    train_losses.append(epoch_train_loss / len(train_loader))

    # nu gaan we evalueren op de validatieset
    # eval-modus zorgt dat lagen zoals dropout niet meer random zijn
    model.eval()
    epoch_val_loss = 0

    # bij validatie willen we geen gradients berekenen
    # dat scheelt tijd en geheugen
    with torch.no_grad():
        for images, labels in val_loader:
            # weer alles naar dezelfde device
            images, labels = images.to(device), labels.to(device)

            # forward pass
            outputs = model(images)

            # loss berekenen
            loss = criterion(outputs, labels)

            # optellen zodat we een gemiddelde kunnen nemen
            epoch_val_loss += loss.item()

    # gemiddelde validatie loss voor deze epoch
    val_losses.append(epoch_val_loss / len(val_loader))

    # korte samenvatting na elke epoch
    print(
        f"--- Epoch {epoch+1} Summary: Train Loss {train_losses[-1]:.4f}, Val Loss {val_losses[-1]:.4f} ---"
    )

# na training maken we een plot zodat je de curve kunt zien
plt.figure(figsize=(10, 5))

# plot de training losses per epoch
plt.plot(range(1, 4), train_losses, label="Train Loss", marker="o")

# plot de validatie losses per epoch
plt.plot(range(1, 4), val_losses, label="Val Loss", marker="o")

# titel en labels zodat de grafiek meteen duidelijk is
plt.title("PCAM Training: First 3 Epochs")
plt.xlabel("Epoch")
plt.ylabel("CrossEntropy Loss")

# legenda zodat je weet welke lijn wat is
plt.legend()

# grid maakt het makkelijker om waardes af te lezen
plt.grid(True)

# we slaan de plot op als png bestand
plt.savefig("pcam_learning_curves.png")

# laatste status print
print("Training complete. Plot saved as pcam_learning_curves.png")