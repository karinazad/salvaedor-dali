import umap.umap_ as umap
import random
import matplotlib.pyplot as plt
import numpy as np


def _create_colormap(labels, group_membership, colors=None):
    """
    Create color handles for legends based on a dictionary.
    Parameteres:
        labels (list):          list or array of the names of different groups
                                e.g: ['IDHwt + Classical','IDHwt + Mesenchymal', 'IDHmut + G-CIMP-high', 'IDHmut + 1p/19q codel', 'Other']
        group_membership (list) list,array or pd.Series that specifies which sample belongs to which group
                                e.g: ['IDHwt + Mesenchymal', 'IDHmut + G-CIMP-high', 'Other', 'Other', 'Other'... ]
    Return:
        colordict (dict):       specifies color coding, used for the legend
                                e.g: {'IDHwt':'mediumblue', IDHmut-non-codel':'tomato', 'IDHmut-codel':'gold'}

        colormap (ndarray):     shape = (N,1)
                                ndarray that specifies color for each sample according to the color_dict
                                used for actual coloring of the points
                                (e.g: 'mediumblue', 'tomato, 'tomato', 'gold, 'tomato', ...)
    """
    n = len(labels)
    if colors is None:
        colors =  ["#" + "%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
    colordict = {}
    for i in range(n):
        colordict[labels[i]] = colors[i]
    colormap = np.vectorize(colordict.get)(np.array(group_membership)).tolist()
    return colormap, colordict


def _create_handles(dctn):
    """
    Create color handles for legends based on a dictionary.
    Parameteres:
        diction (dictionary):   dictionary that was used to created mapping
    Return:
        handles (list):         list of handles to be used in the legend
    """
    import matplotlib.patches as mpatches

    handles = []
    for key in dctn.keys():
        handle = mpatches.Patch(color=dctn[key], label=key)
        handles.append(handle)
    return handles



def plot_umap(data, labels, n_neighbors = 50, min_dist = 0.2,n_components = 2 ):
    metric = 'euclidean'

    fit = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    embedding = fit.fit_transform(data)

    unique_labels = np.unique(labels)
    colormap, colordict = _create_colormap(unique_labels, labels)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=colormap, s=10)
    handle = _create_handles(colordict)
    plt.legend(handles=handle, loc='best')
