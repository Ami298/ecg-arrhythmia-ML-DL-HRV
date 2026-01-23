# ================== IMPORTS ==================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pywt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree, metrics
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
import wfdb
import seaborn as sns

# ================== SETTINGS ==================
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.grid'] = True

# ================== USER PARAMETERS ==================
DATA_PATH = "/home/ajay/Desktop/amish/mit-bih-arrhythmia-database-1.0.0"
WINDOW_SIZE = 180
CLASSES = ['N', 'L', 'R', 'A', 'V']
MAX_COUNT = 10000

# ================== FUNCTIONS ==================
def denoise_signal(signal, wavelet='sym4', threshold=0.04):
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * np.max(coeffs[i]))
    return pywt.waverec(coeffs, wavelet)

def segment_beats(signal, r_peaks, window_size=WINDOW_SIZE):
    half = window_size // 2
    segments = []
    for r in r_peaks:
        if r - half >= 0 and r + half < len(signal):
            segments.append(signal[r-half:r+half])
    return np.array(segments)

def load_data_wfdb(data_path):
    X_all, y_all = [], []
    CLASS_MAP = {'N':0, 'L':1, 'R':2, 'A':3, 'V':4}

    for file in os.listdir(data_path):
        if not file.endswith('.hea'):
            continue
        name = file.replace('.hea','')
        path = os.path.join(data_path, name)

        try:
            record = wfdb.rdrecord(path)
            lead = record.sig_name.index('MLII') if 'MLII' in record.sig_name else 0
            signal = stats.zscore(denoise_signal(record.p_signal[:, lead]))

            r_peaks = np.where(
                (signal[1:-1] > signal[:-2]) &
                (signal[1:-1] > signal[2:]) &
                (signal[1:-1] > 0.5)
            )[0] + 1

            segments = segment_beats(signal, r_peaks)
            ann = wfdb.rdann(path, 'atr')

            for i, r in enumerate(r_peaks[:len(segments)]):
                if r in ann.sample:
                    idx = list(ann.sample).index(r)
                    sym = ann.symbol[idx]
                    if sym in CLASS_MAP:
                        X_all.append(segments[i])
                        y_all.append(CLASS_MAP[sym])

        except Exception as e:
            print(f"❌ Error with record {name}: {e}")

    return np.array(X_all), np.array(y_all)

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

def plot_confusion(y_true, y_pred, title):
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASSES, yticklabels=CLASSES,
                cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"\n================ {name} =================")
    print(f"Accuracy : {acc:.4f}")
    print(metrics.classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))
    plot_confusion(y_test, y_pred, name + " Confusion Matrix")
    return acc

def cross_validate_only(model, X, y, name, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    print(f"\n===== 5-Fold CV: {name} =====")
    print(f"Fold Accuracies : {np.round(scores,4)}")
    print(f"Mean Accuracy   : {scores.mean():.4f}")
    print(f"Std Deviation   : {scores.std():.4f}")
    return scores.mean()

# ================== MAIN ==================
if __name__ == "__main__":

    X, y = load_data_wfdb(DATA_PATH)
    X, y = balance_classes(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train = X_train.reshape(len(X_train), -1)
    X_test  = X_test.reshape(len(X_test), -1)

    model_accuracies = {}

    model_accuracies["KNN"] = evaluate_model(
        KNeighborsClassifier(n_neighbors=7, weights='distance'),
        X_train, y_train, X_test, y_test, "KNN"
    )

    model_accuracies["SVC"] = evaluate_model(
        SVC(kernel='rbf', C=25, gamma=0.01, class_weight='balanced'),
        X_train, y_train, X_test, y_test, "SVC"
    )

    model_accuracies["Random Forest"] = evaluate_model(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_train, y_train, X_test, y_test, "Random Forest"
    )

    model_accuracies["GaussianNB"] = evaluate_model(
        GaussianNB(var_smoothing=1e-8),
        X_train, y_train, X_test, y_test, "GaussianNB"
    )

    model_accuracies["Decision Tree"] = evaluate_model(
        tree.DecisionTreeClassifier(random_state=0),
        X_train, y_train, X_test, y_test, "Decision Tree"
    )

    # ================== CROSS VALIDATION (ADDED) ==================
    print("\n\n================= CROSS VALIDATION RESULTS =================")
    X_flat = X.reshape(len(X), -1)

    cross_validate_only(KNeighborsClassifier(n_neighbors=7, weights='distance'), X_flat, y, "KNN")
    cross_validate_only(SVC(kernel='rbf', C=25, gamma=0.01, class_weight='balanced'), X_flat, y, "SVC")
    cross_validate_only(RandomForestClassifier(n_estimators=100, random_state=42), X_flat, y, "Random Forest")
    cross_validate_only(GaussianNB(var_smoothing=1e-8), X_flat, y, "GaussianNB")
    cross_validate_only(tree.DecisionTreeClassifier(random_state=0), X_flat, y, "Decision Tree")
