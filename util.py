import seaborn as sns
from matplotlib import pyplot as plt

def accuracy_swarmplot(data, x, old_metrics=False, xlabel=None, baseline=1/3, y_label='Mean accuracy', y='accuracy'):
    sns.swarmplot(x=x, y=y, data=data)
    
    plt.ylim([0,1])
    
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.ylabel(y_label)
    plt.axhline(y=baseline, label="baseline")
    
    plt.legend()
    plt.show()