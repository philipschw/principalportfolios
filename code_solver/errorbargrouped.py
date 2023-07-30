# import packages
import numpy as np
import matplotlib.pyplot as plt

def errorbargrouped(data: np.ndarray, err: np.ndarray, numSE: float, *args) -> plt.Figure:
    """
    Create a grouped bar plot with error bars.

    :param data: The data matrix of shape (ngroups, nbars).
    :type data: numpy.ndarray
    :param err: The error matrix of shape (ngroups, nbars).
    :type err: numpy.ndarray
    :param numSE: Number of standard errors for error bars.
    :type numSE: float
    :param *args: Additional arguments.
    :return: The generated matplotlib Figure object.
    :rtype: matplotlib.figureFigure
    """

    # Create a figure and axes object
    fig, ax = plt.subplots(figsize=(12.8,10))

    # Determine the number of groups and bars in each group
    ngroups, nbars = data.shape

    # Use matlab colors
    colors = [
        (0, 0.4470, 0.7410),   # Blue
        (0.8500, 0.3250, 0.0980),   # Orange
        (0.9290, 0.6940, 0.1250),   # Yellow
        (0.4940, 0.1840, 0.5560),   # Purple
        (0.5, 0.5, 0.5),   # Grey
        (0.4660, 0.6740, 0.1880),   # Green
        (0.3010, 0.7450, 0.9330),   # Light Blue
        (0.6350, 0.0780, 0.1840)    # Red
    ]

    # The width of a single bar
    bar_width = 0.8 / nbars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    for i in range(nbars):
    # The offset in x direction of that bar
        x_offset = (i - nbars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for j in range(ngroups):
            bar = ax.bar(j + x_offset, data[j, i], width=bar_width * 0.9, color=colors[i % len(colors)], edgecolor='black')

            # Add a handle to the last drawn bar, which we'll need for the legend
            bars.append(bar[0])

            # Add error bars using the error values
            ax.errorbar(j + x_offset, data[j, i], yerr=numSE * err[j, i], fmt='none', color='black', capsize=2, linewidth=1)

            # Add values on top of each bar
            if args and args[0]:
                if data[j, i] >= 0:
                    ax.text(j + x_offset, data[j, i], f'{data[j, i]:.2f}', ha='center', va='bottom', fontsize=10)
                else:
                    ax.text(j + x_offset, data[j, i], f'{data[j, i]:.2f}', ha='center', va='top', fontsize=10)


    # Set x-axis ticks and labels
    plt.xticks(range(ngroups), list(range(ngroups)))

    # Add horizontal x-axis
    ax.axhline(y=0, color='black', linewidth=1)

    # Include a negative y-axis
    plt.ylim(min(-np.ma.min(np.abs(np.ma.masked_array(data, mask=(data == 0)))), np.min(data-numSE*err))*1.2, np.max(data+numSE*err)*1.2)

    return fig
