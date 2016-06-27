# Create a confusion_matrix graphic

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(['Non-Squiggle', 'Squiggle']))
    plt.xticks(tick_marks, ['Non-Squiggle', 'Squiggle'], rotation=45)
    plt.yticks(tick_marks, ['Non-Squiggle', 'Squiggle'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cm = np.array([[769, 7],[0,73]])

# Normalize the confusion matrix by row (i.e by the number of samples in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Bagging\nNormalized Confusion Matrix on Test Set')

plt.show()
