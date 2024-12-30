import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PlotDistribution:
    def __init__(self, x_values: np.array, samples: np.array):
        self.samples = np.asarray(samples).real
        self.x_values = x_values

    def median_distrib(self) -> np.ndarray:
        return np.median(self.samples, axis=0)

    def perc_distrib(self, perc: int) -> np.ndarray:
        return np.percentile(self.samples, perc, axis=0)


    def save_to_csv(self, file_path: str, lower_perc=25, upper_perc=75, extra_values=None):
        
        data = {
            'x_values': self.x_values,
            'median': self.median_distrib(),
            f'percentile_{lower_perc}': self.perc_distrib(lower_perc),
            f'percentile_{upper_perc}': self.perc_distrib(upper_perc)
        }
        if extra_values is not None:
            data['extra_values'] = extra_values
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

    def plot_estad(self,
                   median_color='blue',
                   fill_color='lightblue',
                   lower_perc=25,
                   upper_perc=75,
                   title=None,
                   xlabel=None,
                   ylabel=None,
                   label='Median empirical distribution',
                   extra_values=None,
                   extra_label='Real distribution',
                   extra_color='orange',
                   extra_style=None,  
                   show_legend=False,
                   ax=None,
                   linewidth=1, 
                   alpha=0.5):  
        if ax is None:
            ax = plt.gca()  
        
        ax.plot(self.x_values, self.median_distrib(), color=median_color, label=label)
        ax.fill_between(self.x_values,
                        self.perc_distrib(lower_perc),
                        self.perc_distrib(upper_perc),
                        color=fill_color, alpha=0.5)

        if extra_values is not None:
            if extra_style is None:
                ax.plot(self.x_values, extra_values, label=extra_label, color=extra_color) 
            else:
                ax.plot(self.x_values, extra_values, label=extra_label, color=extra_color, **extra_style)  

        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if show_legend:
            ax.legend()
 