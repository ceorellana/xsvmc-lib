import matplotlib.widgets
import matplotlib.patches
import mpl_toolkits.axes_grid1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from math import ceil

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

    def mu_nu_chart(self, title, samples_per_page, mu_color, nu_color,show_legend):
        ind, mu_values, nu_values, labels = self.get_ifs_values()
        nu_values = [-nu for nu in nu_values]

        fig, ax = plt.subplots()

        p1 = ax.bar(ind, mu_values, 0.35, label='mu', color=mu_color, edgecolor="black")
        p2 = ax.bar(ind, nu_values, 0.35, label='nu', color=nu_color, edgecolor="black")

        for i in range(len(ind)):
            img = self.pro_misv_pics[i]
            img = OffsetImage(img, zoom=self.zoom)
            img.image.axes = ax

            ab = AnnotationBbox(img, (i, mu_values[i]),  xybox=(0., -25.), frameon=False,
                xycoords='data',  boxcoords="offset points", pad=0)

            ax.add_artist(ab)

            img = self.ifs_pics[i]
            img = OffsetImage(img, zoom=self.zoom)
            img.image.axes = ax

            ab = AnnotationBbox(img, (i, 0),  xybox=(0., 0.), frameon=False,
                xycoords='data',  boxcoords="offset points", pad=0)

            ax.add_artist(ab)

            img = self.con_misv_pics[i]
            img = OffsetImage(img, zoom=self.zoom)
            img.image.axes = ax

            ab = AnnotationBbox(img, (i, nu_values[i]),  xybox=(0., 25.), frameon=False,
                xycoords='data',  boxcoords="offset points", pad=0)

            ax.add_artist(ab)

        nu_values = [round(-nu, 5) for nu in nu_values]

        ax.axhline(0, color='grey', linewidth=0.8)
        if(title==None):
            title = "Membership to class %s" % self.class_id
        ax.set_title(title)
        ax.set_xticks(ind)
        ax.set_xticklabels(labels)
        plt.yticks([])

        if(show_legend):
            ax.legend()

        # Label mu and nu values
        ax.bar_label(p1, label_type='center')
        ax.bar_label(p2, labels=nu_values, label_type='center')

        plt.text(0.09, 0.6, r'$\mu_A$', fontsize=20, transform=plt.gcf().transFigure)
        plt.text(0.91, 0.3, r'$\nu_A$', fontsize=20, transform=plt.gcf().transFigure)

        plt.xlim([-0.5, len(labels)])

        if(samples_per_page is not None):
            if(isinstance(samples_per_page,int)==False or samples_per_page<=0):
                raise ValueError("Value provided must be a positive integer")
            pages = ceil(len(labels)/samples_per_page)
            ax_slider = fig.add_axes([0.1, 0.01, 0.8, 0.04])
            slider = PageSlider(ax_slider, 'Page', pages, activecolor="orange")

        plt.show()
        

    def buoyancy_chart(self, title, samples_per_page, mu_color, nu_color,show_legend):
        ind, mu_values, nu_values, buoyancy_values, labels = self.get_ifs_values(True)
        nu_values = [-nu for nu in nu_values]

        fig, ax = plt.subplots()

        p1 = ax.bar(ind, mu_values, 0.35, label='mu', color=mu_color, edgecolor="black")
        p2 = ax.bar(ind, nu_values, 0.35, label='nu', color=nu_color, edgecolor="black")
        p3 = ax.plot(ind, buoyancy_values, color='blue', label='buoyancy', marker=".", markersize=10, ls="--")

        for i in range(len(ind)):
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
        ax.set_xticks(ind)
        ax.set_xticklabels(labels)
        plt.yticks([])

        if(show_legend):
            ax.legend()

        # Label with label_type 'center' instead of the default 'edge'
        line = ax.lines[0]
        for x_value, y_value in zip(line.get_xdata(), line.get_ydata()):
            label = "{:.2f}".format(y_value)
            y_padding=10
            if(y_value<0):
                y_padding=-10
            ax.annotate(label,(x_value, y_value), 
                textcoords="offset points", xytext=(0,y_padding), ha='center')

        plt.text(0.09, 0.6, r'$\mu_A$', fontsize=20, transform=plt.gcf().transFigure)
        plt.text(0.91, 0.3, r'$\nu_A$', fontsize=20, transform=plt.gcf().transFigure)
        
        if(samples_per_page is not None):
            if(isinstance(samples_per_page,int)==False or samples_per_page<=0):
                raise ValueError("Value provided must be a positive integer")
            pages = ceil(len(labels)/samples_per_page)
            ax_slider = fig.add_axes([0.1, 0.01, 0.8, 0.04])
            slider = PageSlider(ax_slider, 'Page', pages, activecolor="orange")
            fig.subplots_adjust(bottom=0.15)

        plt.show()

    def plot(self, version="buoyancy_values", title=None, samples_per_page=None, mu_color="white", nu_color="gray", show_legend=True):
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

        """
        chart_index = CHART_TYPES.index(version)

        if(chart_index==0):
            self.mu_nu_chart(title, samples_per_page, mu_color, nu_color, show_legend)
        elif(chart_index==1):
            self.buoyancy_chart(title, samples_per_page, mu_color, nu_color, show_legend)