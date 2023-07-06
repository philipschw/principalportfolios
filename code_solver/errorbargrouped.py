import numpy as np
import matplotlib.pyplot as plt

def errorbargrouped(data, err, numSE, *args):
    b = plt.bar(range(data.shape[1]), data, width=0.8)

    # Find the number of groups and the number of bars in each group
    ngroups, nbars = data.shape

    # Calculate the width for each bar group
    groupwidth = min(0.8, nbars / (nbars + 1.5))

    # Set the position of each error bar in the centre of the main bar
    for i in range(nbars):
        # Calculate center of each bar
        if ngroups == 1:
            x = i
        else:
            x = np.arange(ngroups) - groupwidth / 2 + (2 * i - 1) * groupwidth / (2 * nbars)
        plt.errorbar(x, data[:, i], numSE * err[:, i], color='k', linestyle='none')

        if len(args) > 0:
            if i == 0 or ngroups > 1:
                for j, y in enumerate(data[:, i]):
                    plt.text(b[i].get_x() + b[i].get_width() / 2, y, f'{y:.2f}',
                             verticalalignment='bottom', horizontalalignment='center', fontname='timesnewroman',
                             fontsize=args[0])
                    if y < 0:
                        plt.gca().texts[-1].set_verticalalignment('top')

    return b
