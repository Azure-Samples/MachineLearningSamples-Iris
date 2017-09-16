try:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from scipy import interp
    from itertools import cycle
    import numpy as np

    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import label_binarize
except ImportError:
    print("Library (matplotlib or sklearn) missing. Can't plot.")
    

# note this is alphabetically ordered.
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

def plot_iris(clf, X, Y):
    plot_confusion_matrix(clf, X, Y)
    plot_roc(clf, X, Y)

def plot_confusion_matrix(clf, iris_X, iris_Y):
    print("Plotting confusion matrix...")
    # score the entire test set
    Y_hat = clf.predict(iris_X)

    # create a confusion matrix    
    cm = confusion_matrix(iris_Y, Y_hat, labels)

    # plot the confusion matrix
    print("Confusion matrix in text:")
    print(cm)

    fig = plt.figure(figsize=(6, 4), dpi=75)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted Species")
    plt.ylabel("True Species")
    fig.savefig('./outputs/cm.png', bbox_inches='tight')

    print("Confusion matrix plotted.")
    
def plot_roc(clf, iris_X, iris_Y):
    print("Plotting ROC curve....")
    n_classes = len(set(iris_Y))
    Y_score = clf.decision_function(iris_X)
    Y_onehot = label_binarize(iris_Y, classes = labels)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_onehot[:,i], Y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_onehot.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig = plt.figure(figsize=(6, 5), dpi=75)
    # set lineweight
    lw = 2

    # plot micro average
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    # plot macro average
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # plot ROC for each class
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):    
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(labels[i], roc_auc[i]))

    # plot diagnal line
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Iris multi-class ROC')
    plt.legend(loc="lower right")
    fig.savefig("./outputs/roc.png", bbox_inches='tight')

    print("ROC curve plotted.")
