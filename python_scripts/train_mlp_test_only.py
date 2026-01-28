import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- Dataset ---
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_file):
        with open(embeddings_file, "rb") as f:
            self.embeddings = pickle.load(f)
        self.pids = list(self.embeddings.keys())

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        X = torch.tensor(self.embeddings[pid], dtype=torch.float32)
        y = -1  # placeholder voor test-only (geen labels)
        return X, y

# --- MLP model ---
class MLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512,256], num_classes=33, dropout=0.2):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# --- Trainer loop (dummy test run, geen labels) ---
def test_loop(test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    model.eval()
    count = 0
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            out = model(X)  # forward pass
            count += X.size(0)
    print(f"Processed {count} test embeddings")

# --- Usage ---
if __name__=="__main__":
    test_dataset = EmbeddingDataset("data/processed/text/v1/encoder_embeddings_test.pkl")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_loop(test_loader)
