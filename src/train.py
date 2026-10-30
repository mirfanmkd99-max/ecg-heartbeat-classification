import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import f1_score

from data import X_train, y_train, X_val, y_val, X_test, y_test


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_loader(X, y, shuffle):
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)   # (N, 1, 187)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(Xt, yt), batch_size=128, shuffle=shuffle)


train_loader = to_loader(X_train, y_train, True)
val_loader = to_loader(X_val, y_val, False)
test_loader = to_loader(X_test, y_test, False)


class ECGNet(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                        nn.Linear(64, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


model = ECGNet().to(device)
w = compute_class_weight('balanced', classes=np.arange(5), y=y_train)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w,
                                                  dtype=torch.float32).to(device))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


def predict(loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


best_f1 = 0

for epoch in range(20):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    f1 = f1_score(y_val, predict(val_loader), average='macro')

    print(
        f'epoch {epoch:2d}  last batch loss {loss.item():.3f}  val macro F1 {f1:.3f}')
if f1 > best_f1:
    best_f1 = f1
    torch.save(model.state_dict(),
               'results/ecg_cnn_best.pt')
