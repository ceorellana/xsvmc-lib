import matplotlib.widgets
import matplotlib.patches
import mpl_toolkits.axes_grid1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_pdf import PdfPages
import pyx
from math import ceil
import glob
import os

CHART_TYPES = ['mu_nu_values', 'buoyancy_values']

class PageSlider(matplotlib.widgets.Slider):

    def __init__(self, ax, label, numpages = 10, valinit=0, valfmt='%1d', 
                 closedmin=True, closedmax=True,  
                 dragging=True, **kwargs):

        self.facecolor=kwargs.get('facecolor',"w")
        self.activecolor = kwargs.pop('activecolor',"b")
        self.fontsize = kwargs.pop('fontsize', 10)
        self.numpages = numpages

        super(PageSlider, self).__init__(ax, label, 0, numpages, 
                            valinit=valinit, valfmt=valfmt, **kwargs)

        self.poly.set_visible(False)
        self.vline.set_visible(False)
        self.pageRects = []
        for i in range(numpages):
            facecolor = self.activecolor if i==valinit else self.facecolor
            r  = matplotlib.patches.Rectangle((float(i)/numpages, 0), 1./numpages, 1, 
                                transform=ax.transAxes, facecolor=facecolor)
            ax.add_artist(r)
            self.pageRects.append(r)
            ax.text(float(i)/numpages+0.5/numpages, 0.5, str(i+1),  
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=self.fontsize)
        self.valtext.set_visible(False)

        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        fax = divider.append_axes("right", size="5%", pad=0.05)
        self.button_back = matplotlib.widgets.Button(bax, label="<", 
                        color=self.facecolor, hovercolor=self.activecolor)
        self.button_forward = matplotlib.widgets.Button(fax, label=">", 
                        color=self.facecolor, hovercolor=self.activecolor)
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.backward)
        self.button_forward.on_clicked(self.forward)

    def _update(self, event):
        super(PageSlider, self)._update(event)
        i = int(self.val)
        if i >=self.valmax:
            return
        self._colorize(i)

    def _colorize(self, i):
        for j in range(self.numpages):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def forward(self, event):
        current_i = int(self.val)
        i = current_i+1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)

    def backward(self, event):
        current_i = int(self.val)
        i = current_i-1
        if (i < self.valmin) or (i >= self.valmax):
            return
        self.set_val(i)
        self._colorize(i)

class BuoyancyPlot:
    """ Buoyancy representation of a xSVMC

    Implementation based on the article 'Loor, M., Tapia-Rosero, A., & De Tré, G. (2018). Usability of Concordance Indices
    in FAST-GDM Problems. In IJCCI (pp. 67-78).'
    
    Article available for reading here: https://www.scitepress.org/Papers/2018/69565/69565.pdf

    Parameters
    ----------
        
    ifs: list of xAIFSElement of shape(n_test,)
        List of IFSElements given from an Augmented Model

    class_id: int
        Class used to evaluate the elements of ifs.

    ifs_pics: list or ndarray of shape(n_train,), default=None
        List of pictures corresponding to the training set used for clf.fit(). If no
        value is passed, no picture will be added to the graph generated by plot()

    pro_misv_pics: list or ndarray of shape(n_test,), default=None
        List of pictures corresponding to the positive MISV of each IFSElement on ifs. If no
        value is passed, no picture will be added to the graph generated by plot()

    con_misv_pics: list or ndarray of shape(n_test,), default=None
        List of pictures corresponding to the negative MISV of each IFSElement on ifs. If no
        value is passed, no picture will be added to the graph generated by plot()

    zoom: float, default=1.0
        If provided, increases/decreases the pictures size by the provided ratio.

    """

    def __init__(self, ifs, class_id, ifs_pics = None, pro_misv_pics = None, con_misv_pics = None, zoom = 1.0):

        self.ifs = ifs
        self.class_id = class_id
        self.ifs_pics = ifs_pics
        self.pro_misv_pics = pro_misv_pics
        self.con_misv_pics = con_misv_pics
        self.zoom = zoom

    def get_ifs_values(self, buoyancy=False):
        """ Generate mu, nu and buoyance values for each IFS """
        ind = np.arange(len(self.ifs))
        mu_values = []
        nu_values = []
        buoyancy_values = []
        labels = []

        for i,ifs_element in enumerate(self.ifs):
            mu_values.append(ifs_element.mu_hat.value)
            nu_values.append(ifs_element.nu_hat.value)
            buoyancy_values.append(ifs_element.buoyancy)
            labels.append('X%s' % i)

        if(buoyancy):
            return ind, mu_values, nu_values, buoyancy_values, labels
        return ind, mu_values, nu_values, labels

    def chart_values(self, title, mu_color, nu_color, show_legend, nu_hatch, start, end, samples_per_page, bar_align, ax, buoyancy):
        if(buoyancy):
            ind, mu_values, nu_values, buoyancy_values, labels = self.get_ifs_values(buoyancy)
        else:
            ind, mu_values, nu_values, labels = self.get_ifs_values(buoyancy)
        nu_values = [-nu for nu in nu_values]

        p1 = ax.bar(ind[start:end], mu_values[start:end], 0.35, label='mu', color=mu_color, edgecolor="black")
        p2 = ax.bar(ind[start:end], nu_values[start:end], 0.35, label='nu', color=nu_color, edgecolor="black", hatch=nu_hatch)
        if(buoyancy):
            p3 = ax.plot(ind[start:end], buoyancy_values[start:end], color='blue', label='buoyancy', marker=".", markersize=10, ls="--")

        for i in range(start,end):
            # Plot positive MISV picture
            img = self.pro_misv_pics[i]
            img = OffsetImage(img, zoom=self.zoom)
            img.image.axes = ax

            ab = AnnotationBbox(img, (i, mu_values[i]),  xybox=(0., -25.), frameon=False,
                xycoords='data',  boxcoords="offset points", pad=0)

            ax.add_artist(ab)

            # Plot testing sample picture
            img = self.ifs_pics[i]
            img = OffsetImage(img, zoom=self.zoom)
            img.image.axes = ax

            ab = AnnotationBbox(img, (i, 0),  xybox=(40., -200.), frameon=False,
                xycoords='data',  boxcoords="offset points", pad=0)

            ax.add_artist(ab)

            # Plot negative MISV picture
            img = self.con_misv_pics[i]
            img = OffsetImage(img, zoom=self.zoom)
            img.image.axes = ax

            ab = AnnotationBbox(img, (i, nu_values[i]),  xybox=(0., 25.), frameon=False,
                xycoords='data',  boxcoords="offset points", pad=0)

            ax.add_artist(ab)

        ax.axhline(0, color='grey', linewidth=0.8)
        if(title==None):
            title = "Membership to class %s" % self.class_id
        ax.set_title(title)
        ax.set_xticks(ind[start:end])
        ax.set_xticklabels(labels[start:end])
        ax.set_yticks([])

        if(show_legend):
            ax.legend()

        if(buoyancy):
            # Draw buoyancy values
            line = ax.lines[0]
            for x_value, y_value in zip(line.get_xdata(), line.get_ydata()):
                label = "{:.2f}".format(y_value)
                y_padding=10
                if(y_value<0):
                    y_padding=-10
                ax.annotate(label,(x_value, y_value), 
                    textcoords="offset points", xytext=(0,y_padding), ha='center')
        else:
            # Label mu and nu values
            nu_values = [round(-nu, 5) for nu in nu_values]
            ax.bar_label(p1, label_type='center')
            ax.bar_label(p2, labels=nu_values[start:end], label_type='center')

        if (bar_align=='left'):
            xlim_start = ind[start]
            xlim_end = ind[start]+samples_per_page-1 if ind[end-1] == len(self.ifs)-1 and len(self.ifs)%samples_per_page!=0 else ind[end-1]
        elif (bar_align=='center'):
            xlim_start = ind[start]-(samples_per_page-(len(ind[start:end])))/2 if ind[end-1] == len(self.ifs)-1 and len(self.ifs)%samples_per_page!=0 else ind[start]
            xlim_end = ind[end-1]+(samples_per_page-(len(ind[start:end])))/2 if ind[end-1] == len(self.ifs)-1 and len(self.ifs)%samples_per_page!=0 else ind[end-1]

        ax.set_xlim(xlim_start-.5, xlim_end+.5)

        ax.text(0.07, 0.6, r'$\mu_A$', fontsize=20, transform=plt.gcf().transFigure)
        ax.text(0.91, 0.3, r'$\nu_A$', fontsize=20, transform=plt.gcf().transFigure)        

    def buoyancy_chart(self, title, samples_per_page, mu_color, nu_color, show_legend, nu_hatch, bar_align, buoyancy):
        test_size = len(self.ifs)
        start = 0
        end = test_size if samples_per_page is None else samples_per_page

        fig, ax = plt.subplots()
        self.chart_values(title, mu_color, nu_color, show_legend, nu_hatch, start, end, samples_per_page, bar_align, ax, buoyancy=buoyancy)
        
        if(samples_per_page is not None):                
            pages = ceil(test_size/samples_per_page)
            ax_slider = fig.add_axes([0.1, 0.01, 0.8, 0.04])
            slider = PageSlider(ax_slider, 'Page', pages, activecolor="orange")
            fig.subplots_adjust(bottom=0.15)

            def update(val):
                i = int(slider.val)
                start = samples_per_page*i
                end = start + samples_per_page

                if (end > test_size):
                    end = test_size

                ax.clear()
                self.chart_values(title, mu_color, nu_color, show_legend, nu_hatch, start, end, samples_per_page, bar_align, ax, buoyancy=buoyancy)
                fig.canvas.draw()
                fig.canvas.flush_events()

            slider.on_changed(update)

        plt.show()

    def generate_charts(self, filename, title, samples_per_page, mu_color, nu_color, show_legend, nu_hatch, bar_align, buoyancy):
        test_size = len(self.ifs)
        file_extension = filename.split(".")[-1]
        if(file_extension=="pdf"):
            pp = PdfPages(filename)
        if(samples_per_page is None):
            num_graphs = 1
            samples_per_page = test_size
        else:
            num_graphs = ceil(test_size/samples_per_page)
        for i in range(num_graphs):
            start = i*samples_per_page
            end = start + samples_per_page

            if (end > test_size):
                    end = test_size

            fig, ax = plt.subplots()
            self.chart_values(title, mu_color, nu_color, show_legend, nu_hatch, start, end, samples_per_page, bar_align, ax, buoyancy=buoyancy)

            if(file_extension=="pdf"):
                pp.savefig(fig)
            elif(file_extension=="eps"):
                plt.savefig("temp/chart_%s.eps" % i, format='eps')
        if(file_extension=="pdf"):
            pp.close()
            os.system(filename)
        elif(file_extension=="eps"):
            c = pyx.canvas.canvas()
            all_files = glob.glob("temp/*.eps")
            for i, file in enumerate(all_files):
                c.insert(pyx.epsfile.epsfile(0, - i*12, file))
            c.writeEPSfile(filename)
            for file in os.listdir('temp'):
                os.remove(os.path.join('temp', file))


    def plot(self, version="buoyancy_values", title=None, samples_per_page=None, mu_color="white", nu_color="lightgray", show_legend=True, nu_hatch="", bar_align='left'):
        """ Plots the membership representation of the xAIFSElements.

        Parameters
        version: {'mu_nu_values', 'buoyancy_values'}, default='buoyancy_values'
            Specifies the plot version to be used for the graphic representation. It must be one of 'mu_nu_values', or
            'buoyancy_values'. If none is given, 'buoyancy_values' will be used.

        title: string, default=None
            Title for the plot representation. If none is given, the title will default according to the class been
            evaluated for membership.

        samples_per_page: int, default=None
            Number of xAIFSElements that appear per plot. If none is given, the graphic representation won't include
            paging.

        mu_color: string, default="white"
            Color used for the mu values bars in the graphic representation. If none is given, 'white' will be used. The
            color string format corresponds to the matplotlib Color formats available.

        nu_color: string, default="gray"
            Color used for the nu values bars in the graphic representation. If none is given, 'gray' will be used. The
            color string format corresponds to the matplotlib Color formats available.

        show_legend: boolean, default=True
            Enable or disable value legend on the plot. If none is given, defaults to True.

        nu_hatch: string, default=""
            Add hatching pattern to nu bars. If none is given, no pattern will be used. Available patterns correspond to the
            matplotlib 'hatch' patterns and format. See https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_hatch
            for a full list of patterns.

        bar_align: {'left', 'center'}, default='left'
            Alignment applied to bars for last page if number of IFS do not correspond to the 'samples_per_page' value.
            If none is given, 'left' will be used.

        """
        if (samples_per_page is not None):
            if(isinstance(samples_per_page,int)==False or samples_per_page<=0):
                raise ValueError("Value provided must be a positive integer")
        
        chart_index = CHART_TYPES.index(version)

        if(chart_index==0):
            self.buoyancy_chart(title, samples_per_page, mu_color, nu_color, show_legend, nu_hatch, bar_align, False)
        elif(chart_index==1):
            self.buoyancy_chart(title, samples_per_page, mu_color, nu_color, show_legend, nu_hatch, bar_align, True)

    def save_plot(self, filename, version="buoyancy_values", title=None, samples_per_page=None, mu_color="white", nu_color="lightgray", show_legend=True, nu_hatch="", bar_align='left'):
        """ Save plot membership representation of the xAIFSElements.

        Parameters (same as plot method save 'filename')
        filename: string
            Output file path/name for the plot. Currently supported formats: pdf, eps

        """
        if (samples_per_page is not None):
            if(isinstance(samples_per_page,int)==False or samples_per_page<=0):
                raise ValueError("Value provided must be a positive integer")
        
        chart_index = CHART_TYPES.index(version)

        if(chart_index==0):
            self.generate_charts(filename, title, samples_per_page, mu_color, nu_color, show_legend, nu_hatch, bar_align, False)
        elif(chart_index==1):
            self.generate_charts(filename, title, samples_per_page, mu_color, nu_color, show_legend, nu_hatch, bar_align, True)