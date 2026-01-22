# ================== IMPORTS ==================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import pywt
from scipy import stats
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ================== SETTINGS ==================
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.grid'] = True
sns.set_style('darkgrid')

# ================== PARAMETERS ==================
DATA_PATH = "/home/ajay/Desktop/amish/mit-bih-arrhythmia-database-1.0.0"
WINDOW_SIZE = 180
CLASSES = ['N', 'L', 'R', 'A', 'V']
BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== EARLY STOPPING ==================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None
        self.stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore(self, model):
        model.load_state_dict(self.best_state)

# ================== FUNCTIONS ==================
def denoise_signal(signal, wavelet='sym4', threshold=0.04):
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * np.max(coeffs[i]))
    return pywt.waverec(coeffs, wavelet)

def segment_beats(signal, r_peaks, window_size):
    half = window_size
    segments = []
    for r in r_peaks:
        if r - half >= 0 and r + half < len(signal):
            segments.append(signal[r - half:r + half])
    return np.array(segments)

def load_data_wfdb(data_path):
    X, y = [], []
    CLASS_MAP = {'N':0, 'L':1, 'R':2, 'A':3, 'V':4}

    for file in os.listdir(data_path):
        if not file.endswith('.hea'):
            continue
        record_name = file.replace('.hea','')
        path = os.path.join(data_path, record_name)

        try:
            record = wfdb.rdrecord(path)
            lead = record.sig_name.index('MLII') if 'MLII' in record.sig_name else 0
            signal = record.p_signal[:, lead]
            signal = stats.zscore(denoise_signal(signal))

            r_peaks = np.where(
                (signal[1:-1] > signal[:-2]) &
                (signal[1:-1] > signal[2:]) &
                (signal[1:-1] > 0.5)
            )[0] + 1

            segments = segment_beats(signal, r_peaks, WINDOW_SIZE)
            ann = wfdb.rdann(path, 'atr')

            for i, r in enumerate(r_peaks[:len(segments)]):
                if r in ann.sample:
                    idx = list(ann.sample).index(r)
                    sym = ann.symbol[idx]
                    if sym in CLASS_MAP:
                        X.append(segments[i])
                        y.append(CLASS_MAP[sym])

        except Exception as e:
            print(f"Error loading {record_name}: {e}")

    return np.array(X), np.array(y)

def balance_classes(X, y, n_samples=5000):
    df = pd.DataFrame(X)
    df['label'] = y
    balanced = []
    for i in range(len(CLASSES)):
        df_i = df[df['label'] == i]
        df_i = resample(df_i, replace=True, n_samples=n_samples, random_state=42)
        balanced.append(df_i)
    df = pd.concat(balanced).sample(frac=1, random_state=42)
    return df.iloc[:, :-1].values, df['label'].values

# ================== MODEL ==================
class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 13, padding='same'), nn.ReLU(), nn.AvgPool1d(3,2),
            nn.Conv1d(16, 32, 15, padding='same'), nn.ReLU(), nn.AvgPool1d(3,2),
            nn.Conv1d(32, 64, 17, padding='same'), nn.ReLU(), nn.AvgPool1d(3,2),
            nn.Conv1d(64,128, 19, padding='same'), nn.ReLU(), nn.AvgPool1d(3,2),
            nn.Conv1d(128,256,21, padding='same'), nn.ReLU(), nn.AvgPool1d(3,2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256*10, 35),
            nn.Linear(35, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ================== DATA ==================
X, y = load_data_wfdb(DATA_PATH)
X, y = balance_classes(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_x = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
test_x  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
train_y = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
test_y  = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

train_loader = DataLoader(TensorDataset(train_x, train_y), BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TensorDataset(test_x, test_y), BATCH_SIZE)

# ================== TRAIN ==================
model = CNN1D(len(CLASSES)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
early_stopping = EarlyStopping(patience=10)

train_losses, val_losses, val_accs = [], [], []

for epoch in range(EPOCHS):
    model.train()
    loss_sum = 0

    for x, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), yb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    train_loss = loss_sum / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    correct, total, val_loss = 0, 0, 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for x, yb in test_loader:
            out = model(x)
            val_loss += criterion(out, yb).item()
            preds = torch.argmax(out, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(yb.cpu().numpy())

    val_loss /= len(test_loader)
    acc = correct / total

    val_losses.append(val_loss)
    val_accs.append(acc)

    print(f"Epoch {epoch+1:03d} | Train {train_loss:.4f} | Val {val_loss:.4f} | Acc {acc:.4f}")

    early_stopping(val_loss, model)
    if early_stopping.stop:
        print("Early stopping triggered")
        break

early_stopping.restore(model)

# ================== EVALUATION ==================
print(classification_report(labels_all, preds_all, target_names=CLASSES))

cm = confusion_matrix(labels_all, preds_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.show()

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.show()

plt.plot(val_accs, label='Val Accuracy')
plt.legend()
plt.show()

