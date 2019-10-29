# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Fri Sep 13 16:18:34 2019
"""


# standard imports
import os
import sys
#import pickle as pi
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename

# other imports
import numpy as np
import skimage as ski
from scipy.signal import convolve2d

# matplotlib and settings
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
matplotlib.use("TkAgg")
# WARNING above will not work in idle. must be called from terminal


class TSAgui():
    "Graphical User Interface for utalising the TSA tool set"

    def __init__(self, master=None):

        # initalise tk obj (aka. master)
        master = tk.Tk() if master is None else master
        self.master = master

        # master settings
        master.title('Image and Tabs')
        master.geometry('1000x500+300+300')

        # create image side
        left = ttk.Frame(master)
        self.img_obj = ImageWidgetHandler(left)

        # create tabs side
        right = ttk.Frame(master)
        tab_parent = ttk.Notebook(right)

        tab_one = LoadTab(tab_parent, self)
        tab_parent.add(tab_one, text='Load Images')
        tab_two = SelectTab(tab_parent, self)
        tab_parent.add(tab_two, text='Manual Clustering')

        tab_parent.pack(fill='both', expand=True)

        # create vairables to be populated
        self.clust_dict = None

        # pack and start the main window
        left.pack(side=tk.LEFT, expand=1, fill='both')
        right.pack(side=tk.RIGHT, expand=1, fill='both')
        master.mainloop()


#    def load_cluster_dict(self, path):
#
#        # reset mask
#        self.clust_dict = None
#        err_string = ''
#
#        # if the file exists try to load it
#        if os.path.isfile(path):
#            try:
#                with open(path, 'rb') as handle:
#                    self.clust_dict = pi.load(handle)
#            except:
#                err_string = 'Error when loading cluster dictionary:\n' + \
#                             str(sys.exc_info()[1])
#        else:
#            err_string = 'File not found'
#
#        return err_string


class LoadTab(ttk.Frame):
    "Tab for loading in data from files and svaing data to files"

    def __init__(self, parent, gui):
        "parent of this widget and gui obj need to be passed in"

        # store the parent and master
        self.parent = parent
        self.gui = gui

        # initalise frame
        super().__init__(parent)

        # create load image form
        self._create_entry_widget('Image path', gui.img_obj.load_img)

        # create load mask form
        self._create_entry_widget('Mask path', gui.img_obj.load_mask)


    def _create_entry_widget(self, text, func):
        """
        Create an entry field with text that calls func with given string.
        If func returns a string, this is treated as an error message
        """

        # group holds all items together
        group = tk.LabelFrame(self, text=text, padx=5, pady=5)
        group.pack(padx=5, pady=5, expand=True, fill='x')

        # create the entry form
        ent = tk.Entry(group)
        ent.pack(fill='x', expand=True)

        # create the function to be called
        def func_dummy():
            'calls given func with entered string and displays error message'
            output = func(ent.get())
            if output:
                tk.messagebox.showinfo(title='Error', message=output)

        # create enter button
        but1 = tk.Button(group, text='Load', width=10, command=func_dummy)
        but1.pack(side='left', padx=5, pady=5)
        
        # create browse button
        def browse_dummy():
            'use a file browser to get the filename'
            f_name = askopenfilename(initialdir=os.getcwd())
            ent.delete(0,tk.END)
            ent.insert(0,f_name)
#            func_dummy()
        but2 = tk.Button(group, text='browse', width=10, command=browse_dummy)
        but2.pack(side='right', padx=15, pady=5)


class SelectTab(ttk.Frame):
    "Tab for selecting indevidual segments"

    def __init__(self, parent, gui):
        "parent of this widget and gui obj need to be passed in"

        # store the parent and master
        self.parent = parent
        self.gui = gui

        # initalise frame
        super().__init__(parent)

        # add clear button
        def reset_selection():
            gui.img_obj.selected = []
            gui.img_obj.show()
        but1 = tk.Button(self, text='Clear', width=10, command=reset_selection)
        but1.pack()

        ############### change to selection toggle
        def callback(event):
            'add the selected segment to img' ###############
            lims = gui.img_obj.axs.get_xlim(), gui.img_obj.axs.get_ylim()
            seg_id = gui.img_obj.get_segment(event.xdata, event.ydata)
            gui.img_obj.update_selected(seg_id)
#            gui.img_obj.axs.set(xlim=lims[0], ylim=lims[1])
        
        gui.img_obj.add_function('button_press_event', callback)



class ImageWidgetHandler():
    "Leverage matplotlibs figures to show and interact with images"

    def __init__(self, parent):

        self.parent = parent

        # setup the figure
        self.fig = Figure(dpi=100)
        self.axs = self.fig.add_axes([0.01, 0.01, 0.99, 0.99])
        self.axs.text(.4, .5, s='Please Load an image')
        self.axs.axis('off')

         # create canvas object
        fig_canvas = FigureCanvasTkAgg(self.fig, parent)
        canvas = fig_canvas.get_tk_widget()

        # create the tool bar
        toolbar = NavigationToolbar2Tk(fig_canvas, parent)
        toolbar.update()

        # pack the canvas
        canvas.pack(fill='both', expand=True)

        # create the img and mask vairables
        self.img = None
        self.mask = None
        self.edge_mask = None
        self.selected = []


    def show(self, text='Please Load an image'):
        """
        imshow the current image and mask (if present)
        else print text on canvas
        """

        # clear canvas
        self.axs.clear()
        self.axs.axis('off')

        # plot image
        if self.img is not None:
            self.axs.imshow(self.img)
            # plot mask
            if self.mask is not None:
                self.axs.imshow(self.edge_mask, alpha=0.5)
        # display text if no image present
        else:
            self.axs.text(.4, .5, s=text)

        # update the canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def update_selected(self, seg_id):
        """
        Toggle the given segement in the selected list and highlight those
        selected
        """

        # if in the list remove it
        if seg_id in self.selected:
            self.selected.remove(seg_id)

        # if not in the list add it
        else:
            self.selected.append(seg_id)
            
        print(self.selected)

        self.highlight_segments(self.selected)


    def highlight_segments(self, to_highlight, color='g'):
        'highlight the segments given with the given color'

        # refresh the plot
        self.show()

        # get selected array and overlay it
        select_arr = np.isin(self.mask, to_highlight).astype('float')
        select_arr[select_arr == 0] = np.nan
        print(select_arr)
        self.axs.imshow(select_arr, alpha=0.5)
        
        # update the canvas
        self.fig.canvas.draw()


    def load_img(self, path):
        'Load the image if it exists and call show, returns a string for user'

        # reset image
        self.img = None
        err_string = ''

        # if the file exists try to load it
        if os.path.isfile(path):
            try:
                self.img = ski.io.imread(path)
            except BaseException:
                err_string = 'Error when loading image:\n' + \
                             str(sys.exc_info()[1])
        else:
            err_string = 'File not found'

        # show the new image
        self.show()
        return err_string


    def load_mask(self, path):
        'Load the mask if it exists and call show, returns a string for user'

        # reset mask
        self.mask = None
        err_string = ''

        # if the file exists try to load it
        if os.path.isfile(path):
            try:
                self.mask = np.loadtxt(path)
                self.edge_mask = outline(self.mask)
            except BaseException:
                err_string = 'Error when loading mask:\n' + \
                             str(sys.exc_info()[1])
        else:
            err_string = 'File not found'

        # show the new mask
        self.show()
        return err_string


    def add_function(self, event, func):
        "add function to canvas event"
        self.fig.canvas.mpl_connect(event, func)


    def get_segment(self, x_cord, y_cord):
        "return the segment at cordinate (x, y)"
        return self.mask[int(y_cord), int(x_cord)]


#%%

#import numpy as np
#from scipy.signal import convolve2d

def rgba(mask, color='r', opaqueness=1):
    """
    Take a 2d mask and return a 4d rgba mask for imshow overlaying
    with the given color and opaqueness.
    """

    # validate input
    assert mask.ndim == 2, 'Must be a 2d mask'

    # create the transparent mask
    zeros = np.zeros_like(mask)
    arr = np.dstack([zeros, zeros, zeros, mask*opaqueness])

    # set the correct color channel
    i = ['r', 'g', 'b'].index(color)
    arr[:, :, i] = mask

    return arr


def outline(mask, diag=True, multi=True):
    """
    Take the stored mask and use a laplacian convolution to find the
    outlines for plotting. diag decides if diagonals are to be
    included or not, original decides if the original mask should
    be used or not.

    multi is an option to do horizontal, vertical and multi_directional
    laplacians and combine them. This is a safer method as particular
    geometries can trick the above convolution.
    """
    # select the correcy arrays based on the options given
    lap = np.array([[1., 1., 1.],
                    [1., -8., 1.],
                    [1., 1., 1.]]) if diag else \
          np.array([[0., 1., 0.],
                    [1., -4., 1.],
                    [0., 1., 0.]])

    # do the convolution to find the edges
    conv = convolve2d(mask, lap, mode='valid').astype(bool)

    if multi:
        lap_h = np.array([[1., 2., 1.],
                          [0., 0., 0.],
                          [-1., -2., -1.]])
        lap_v = np.array([[-1., 0., 1.],
                          [-2., 0., 2.],
                          [-1., 0., 1.]])

        conv2 = convolve2d(mask, lap_h, mode='valid').astype(bool)
        conv3 = convolve2d(mask, lap_v, mode='valid').astype(bool)
        conv = (conv + conv2 + conv3).astype(bool)


    # pad back boarders to have same shape as original image
    conv = np.pad(conv, 1, 'edge').astype(float)
    conv[conv == 0] = np.nan

    return conv


#%%

if __name__ == '__main__':
    TSAgui()
