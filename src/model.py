from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, f1_score, ConfusionMatrixDisplay, confusion_matrix

from data import X_train, y_train, X_val, y_val

rf = RandomForestClassifier(
    n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

val_pred = rf.predict(X_val)

cm = confusion_matrix(y_val, val_pred)
print('confusion matrix:\n', cm)

classes = ['N', 'S', 'V', 'F', 'Q']
recall = recall_score(y_val, val_pred, average=None)
f1 = f1_score(y_val, val_pred, average=None)
counts = np.bincount(y_val)

for i in np.argsort(f1):
    print(
        f'{classes[i]}: F1 {f1[i]:.3f}, recall {recall[i]:.3f}, examples {counts[i]}')


print(classification_report(y_val, val_pred, digits=3, target_names=[
      'N normal', 'S supraventricular', 'V ventricular', 'F fusion', 'Q unknown']))
print('macro f1 score: ', f1_score(y_val, val_pred, average='macro'))
ConfusionMatrixDisplay.from_predictions(y_val, val_pred)

ConfusionMatrixDisplay.from_predictions(
    y_val, val_pred, normalize='true', values_format='.2f')
plt.savefig('results/rf_confusion_val_norm.png', dpi=150)

plt.savefig('results/rf_confusion_val.png', dpi=150)
