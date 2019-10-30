# -*- coding: utf-8 -*-
"""
author: Richard Bonye (github Boyne272)
Last updated on Fri Sep 13 16:18:34 2019
"""


# standard imports
import os
import sys
#import ctypes
import pickle as pi
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
        master.geometry('1200x800+100+5')
#        try:
#            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('tsa_logo.ico')
#            master.wm_iconbitmap('tsa_logo.ico')
#        except BaseException:
#            pass

        # create vairables to be populated
        self.clust_dict = None
        self.selected_segs = []

        # create image side
        left = ttk.Frame(master)
        self.img_obj = ImageWidgetHandler(left, self)

        # create tabs side
        right = ttk.Frame(master)
        tab_parent = ttk.Notebook(right)

        self.Load_Tab = LoadTab(tab_parent, self)
        self.Segment_Tab = SegmentTab(tab_parent, self)
        self.Cluster_Tab = ClusterTab(tab_parent, self)
        self.Options_Tab = OptionsTab(tab_parent, self)

        tab_parent.add(self.Load_Tab, text='Load Images')
        tab_parent.add(self.Segment_Tab, text='Segments')
        tab_parent.add(self.Cluster_Tab, text='Clusters')
        tab_parent.add(self.Options_Tab, text='Options')

        tab_parent.pack(fill='both', expand=True)

        # pack and start the main window
        left.pack(side=tk.LEFT, expand=1, fill='both')
        right.pack(side=tk.RIGHT, expand=1, fill='both')
        master.mainloop()


    def update_selected(self, seg_id):
        "Toggle the given segement in the selected list and update the image"

        if seg_id in self.selected_segs:
            self.selected_segs.remove(seg_id)
        else:
            self.selected_segs.append(seg_id)

        self.img_obj.show()


    def error_message(self, string, title='Error'):
        tk.messagebox.showinfo(title=title, message=string)



#%%

class LoadTab(ttk.Frame):
    "Tab for loading in data from files and svaing data to files"

    def __init__(self, parent, gui):
        "parent of this widget and gui obj need to be passed in"

        # store the parent and master
        self.parent = parent
        self.gui = gui

        # initalise frame
        super().__init__(parent)

        # create load forms
        load_group = tk.LabelFrame(self, text='Load', padx=5, pady=5)
        load_group.pack(padx=5, pady=5, expand=True, fill='x')

        self.create_entry_widget(load_group, self._load_img,
                                 title='Image path',
                                 entertext='Load',
                                 search_button=True)
        self.create_entry_widget(load_group, self._load_mask,
                                 title='Mask path',
                                 entertext='Load',
                                 search_button=True)

        tmp_grp = self.create_entry_widget(load_group, self._load_clust,
                                           title='Clustering path',
                                           entertext='Load',
                                           search_button=True)
        but = tk.Button(tmp_grp, text='Make Blank Clustering', width=30,
                        command=self.create_blank_clustering)

        but.pack(side='right', padx=5, pady=10)

        # create the save forms
        save_group = tk.LabelFrame(self, text='Save', padx=5, pady=5)
        save_group.pack(padx=5, pady=5, expand=True, fill='x')

        lab = tk.Label(save_group,
                       text='Please give relative paths without extensions')
        lab.pack(padx=5, pady=5, expand=True, fill='x')

        self.create_entry_widget(save_group, self._save_img,
                                 title='Image path', entertext='Save',
                                 search_button=False)
        self.create_entry_widget(save_group, self._save_mask,
                                 title='Mask path', entertext='Save',
                                 search_button=False)
        self.create_entry_widget(save_group, self._save_clust,
                                 title='Clustering path', entertext='Save',
                                 search_button=False)


    def create_entry_widget(self, parent, func, title='',
                            entertext='Enter', search_button=False):
        """
        Create an entry field that calls func with the inputted string.
        If func returns a string, this is treated as an error message.
        Will also have a file browse button.
        Text is the title for the section
        Returns the group obj for adding more to this section
        """

        # group holds all items together
        group = tk.LabelFrame(parent, text=title, padx=5, pady=5,
                              borderwidth=0)
        group.pack(padx=5, pady=5, expand=True, fill='x')

        # create the entry form
        ent = tk.Entry(group)
        ent.pack(fill='x', expand=True)

        # create browse button
        def browse_dummy():
            'use a file browser to get the filename'
            f_name = askopenfilename(initialdir=os.getcwd())
            ent.delete(0, tk.END)
            ent.insert(0, f_name)
        if search_button:
            but2 = tk.Button(group, text='Search', width=10, command=browse_dummy)
            but2.pack(side='left', padx=15, pady=5)

        # create enter button
        def func_dummy():
            'calls given func with entered string and displays error message'
            output = func(ent.get())
            if output:
                self.gui.error_message(output)
        but1 = tk.Button(group, text=entertext, width=10, command=func_dummy)
        but1.pack(side='left', padx=5, pady=5)

        return group


    def _load_img(self, path):
        'Load the image if it exists and call show, returns a string for user'

        # reset image
        self.gui.img_obj.img = None

        # if the file exists try to load it to img_obj
        if os.path.isfile(path):
            try:
                self.gui.img_obj.img = ski.io.imread(path)/255.
            except BaseException:
                self.gui.error_message('Error when loading image:\n' +
                                       str(sys.exc_info()[1]))
        else:
            self.gui.error_message('File not found')

        # show the new image
        self.gui.img_obj.show()


    def _load_mask(self, path):
        'Load the mask if it exists and call show, returns a string for user'

        # reset mask
        self.gui.img_obj.mask = None

        # if the file exists try to load it to img_obj
        if os.path.isfile(path):
            try:
                self.gui.img_obj.mask = np.loadtxt(path)
                self.gui.img_obj.original_mask = self.gui.img_obj.mask.copy()
                self.gui.img_obj.edge_mask = outline(self.gui.img_obj.mask)
            except BaseException:
                self.gui.error_message('Error when loading mask:\n' +
                                       str(sys.exc_info()[1]))
        else:
            self.gui.error_message('File not found')

        # show the new mask
        self.gui.img_obj.show()


    def _load_clust(self, path):
        'Load the cluster dictionary if it exists'
        # reset mask
        self.gui.clust_dict = None

        # if the file exists try to load it
        if os.path.isfile(path):
            try:
                with open(path, 'rb') as handle:
                    self.gui.clust_dict = pi.load(handle)
            except BaseException:
                self.gui.error_message('Error when loading cluster dict:\n' +
                                       str(sys.exc_info()[1]))
        else:
            self.gui.error_message('File not found')

        # update the cluster selection pannel
        self.gui.Cluster_Tab.refresh_listbox()
        print(self.gui.clust_dict)


    def _save_img(self, path):
        path = path + '.tif'
        self.gui.img_obj.fig.savefig(path, bbox_inches='tight',
                                     pad_inches=0)
        self.gui.error_message(title='Success',
                               string='Image saved as ' + path)


    def _save_mask(self, path):
        path = path + '.txt'
        np.savetxt(path, self.gui.img_obj.mask)
        self.gui.error_message(title='Success',
                               string='mask saved as ' + path)


    def _save_clust(self, path):
        path = path + '.pkl'
        file = open(path, 'wb')
        pi.dump(self.gui.clust_dict, file)
        file.close()
        self.gui.error_message(title='Success',
                               string='clustering saved as ' + path)


    def create_blank_clustering(self):

        # check mask is present
        if self.gui.img_obj.mask is None:
            self.gui.error_message('You must have a mask loaded first')

        # create the dictionary
        self.gui.clust_dict = {'Segment ' + str(int(_id)):[_id] for _id in
                               np.unique(self.gui.img_obj.mask)}

        # update the cluster_menu options
        self.gui.Cluster_Tab.refresh_listbox()



#%%

class ClusterTab(ttk.Frame):
    ""

    def __init__(self, parent, gui):
        "parent of this widget and gui obj need to be passed in"

        # store the parent, master and initalise frame
        self.parent = parent
        self.gui = gui
        super().__init__(parent)

        # intialise attributes
        self.selected_clusters = None

        # create top text
        lab = tk.Label(self, text='Methods to do with clustering')
        lab.pack(padx=10, pady=20, expand=True, fill='x', anchor='n')

        # create pannels
        self._create_cluster_menu()


    def _create_cluster_menu(self):

        # create the containing frames
        frm1 = tk.Frame(self)
        frm1.pack(fill='x', padx=20, pady=20)
        frm2 = tk.Frame(self)
        frm2.pack(fill='x', padx=20, pady=20)

        # create listbox
        self.listbox = tk.Listbox(frm1, selectmode=tk.MULTIPLE)
        self.listbox.bind("<<Button-1>>", self.refresh_listbox)
        self.listbox.pack(side='left', expand=True, fill='x')

        # create the scroll bar
        scrollbar = tk.Scrollbar(frm1, orient=tk.VERTICAL)
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side='right', fill='y', anchor='n')
        self.listbox.configure(yscrollcommand=scrollbar.set)

        # create buttons
        but1 = tk.Button(frm2, text='Update', width=15,
                         command=self.update_selected_clusters)
        but1.pack(side='left', padx=10)

        select_all = lambda: self.listbox.select_set(0, tk.END)
        but2 = tk.Button(frm2, text='Select All', width=15,
                         command=select_all)
        but2.pack(side='left', padx=10)

        clear_all = lambda: self.listbox.select_clear(0, tk.END)
        but3 = tk.Button(frm2, text='Clear', width=15,
                         command=clear_all)
        but3.pack(side='left', padx=10)

        # create listbox entries
        self.refresh_listbox()
        clear_all()


    def refresh_listbox(self, event=None):

        print(type(event))
        
        # clear old entries
        self.listbox.delete(0, tk.END)

        # create entries in cluster loaded
        if self.gui.clust_dict:
            for name in self.gui.clust_dict.keys():
                self.listbox.insert(tk.END, name)

        # output no cluster loaded
        else:
            self.listbox.insert(tk.END, 'No cluster currently loaded')


    def update_selected_clusters(self):

        indexs = np.array(self.listbox.curselection())
        if indexs.size:
            lst = list(self.gui.clust_dict.values())
            self.selected_clusters = np.hstack(np.array(lst)[indexs])
        else:
            self.selected_clusters = None

        self.gui.img_obj.show()



#%%

class OptionsTab(ttk.Frame):
    ""

    def __init__(self, parent, gui):
        "parent of this widget and gui obj need to be passed in"

        # store the parent, master and initalise frame
        self.parent = parent
        self.gui = gui
        super().__init__(parent)

        # make the pannels
        self._create_outline_checkbox()


    def _create_outline_checkbox(self):

        # group holds all items together
        group = tk.LabelFrame(self, text='Options', padx=5, pady=5)
        group.pack(padx=5, pady=20, expand=True, fill='x', anchor='n')

        # add the grid checkbox
        def toggle():
            if see_grid.get():
                self.gui.img_obj.show_edges = True
            else:
                self.gui.img_obj.show_edges = False
            self.gui.img_obj.show()

        see_grid = tk.BooleanVar()
        see_grid.set(True)
        box1 = tk.Checkbutton(group, text='Segment Edges',
                              variable=see_grid, command=toggle)
        box1.pack(padx=5, pady=5)

        return group



#%%


class SegmentTab(ttk.Frame):
    "Tab for selecting indevidual segments"

    def __init__(self, parent, gui):
        "parent of this widget and gui obj need to be passed in"

        # store the parent, master and initalise frame
        self.parent = parent
        self.gui = gui
        super().__init__(parent)

        # create pannels
        self._create_select_checkbox()
        self._create_buttons()


    def _create_select_checkbox(self):
        "allows you to toggle manual selection"

        def select_seg(event):
            "update the selected segments with that clicked"
            seg_id = self.gui.img_obj.get_segment(event.xdata, event.ydata)
            self.gui.update_selected(seg_id)

        def toggle():
            if manual_select.get():
                self.gui.img_obj.add_function('button_press_event', select_seg)
            else:
                self.gui.img_obj.remove_function()

        # add selection toggle
        manual_select = tk.BooleanVar()
        manual_select.set(True)
        toggle()
        box1 = tk.Checkbutton(self, text='Manual Select',
                              variable=manual_select, command=toggle)
        box1.pack(padx=5, pady=5)


    def _create_buttons(self):
        "buttons to merge, unmerge and clear selected"

        # group holds all items together
        group = tk.LabelFrame(self, text='Selection Actions', padx=5, pady=5)
        group.pack(padx=5, pady=5, expand=True, fill='x')

        lab = tk.Label(group, text='Actions to be done on selected segments')
        but1 = tk.Button(group, text='Merge', width=20,
                         command=self.merge_selected)
        but2 = tk.Button(group, text='Un-Merge', width=20,
                         command=self.unmerge_selected)
        but3 = tk.Button(group, text='Clear', width=20,
                         command=self.reset_selection)

        lab.pack(padx=5, pady=5)
        but1.pack(side='left', padx=5, pady=5)
        but2.pack(side='left', padx=5, pady=5)
        but3.pack(side='left', padx=5, pady=5)

        return group


    def merge_selected(self):
        ""

        # if nothing selected error and return
        if not self.gui.selected_segs:
            self.gui.error_message('No segments currently selected')
            return

        # change the mask to the first segments id
        bool_arr = np.isin(self.gui.img_obj.mask, self.gui.selected_segs)
        self.gui.img_obj.mask[bool_arr] = self.gui.selected_segs[0]

        # need to correctly handel cluster dict

        # clear current selection
        self.gui.selected_segs = []
        self.gui.img_obj.show(mask_change=True)


    def unmerge_selected(self):
        ""

        # if nothing selected error and return
        if not self.gui.selected_segs:
            self.gui.error_message('No segments currently selected')
            return

        # change the mask to the original mask in selected regions
        bool_arr = np.isin(self.gui.img_obj.mask, self.gui.selected_segs)
        self.gui.img_obj.mask[bool_arr] = self.gui.img_obj.original_mask[bool_arr]

        # need to correctly handel cluster dict

        # clear current selection
        self.gui.selected_segs = []
        self.gui.img_obj.show(mask_change=True)


    def reset_selection(self):
        self.gui.selected_segs = []
        self.gui.img_obj.show()



#%%


class ImageWidgetHandler():
    "Leverage matplotlibs figures to show and interact with images"


    # options to be set by gui
    edge_cmap = 'Reds_r'
    highlight_rgb = [0., 1., 0.]
    show_edges = True

    def __init__(self, parent, gui):

        # store the parent
        self.parent = parent
        self.gui = gui

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

        # create later needed vairables
        self.img = None
        self.mask = None
        self.original_mask = None
        self.edge_mask = None
        self.canvas_funcs = []


    def show(self, mask_change=False):
        """
        imshow the current image and mask (if present)
        else print text on canvas
        """

        # clear canvas
        self.axs.clear()
        self.axs.axis('off')

        # if no image is loaded give message
        if self.img is None:
            self.axs.text(.4, .5, s='Please Load an image')

        # create the rgb array
        else:
            arr = self.img.copy()

            # if the edge mask has been changed
            if mask_change:
                self.edge_mask = outline(self.mask)

            # if the edges are to be show
            if self.show_edges and self.edge_mask is not None:
                arr[self.edge_mask] = [1., 0., 0.]

            # if clusters are selected
            if self.gui.Cluster_Tab.selected_clusters is not None:
                bool_arr = np.isin(self.mask,
                                   self.gui.Cluster_Tab.selected_clusters)
                arr = anti_highlight(arr, bool_arr)

            # if segments are to be highlighed
            if self.gui.selected_segs:
                arr = highlight(arr,
                                np.isin(self.mask, self.gui.selected_segs),
                                alpha=0.5,
                                color=ImageWidgetHandler.highlight_rgb)

            # plot the array
            self.axs.imshow(arr)

        # update the canvas
        self.fig.canvas.toolbar.forward() # return to zoomed view
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # verbose
#        print('figure redrawn')


    def add_function(self, event, func):
        "add function to canvas event"
        cid = self.fig.canvas.mpl_connect(event, func)
        self.canvas_funcs.append(cid)


    def remove_function(self, index=-1):
        "remove function to canvas event"
        cid = self.canvas_funcs.pop(index)
        self.fig.canvas.mpl_disconnect(cid)


    def get_segment(self, x_cord, y_cord):
        "return the segment at cordinate (x, y)"
        if self.mask is not None:
            return self.mask[int(y_cord), int(x_cord)]



#%%

#import numpy as np
#from scipy.signal import convolve2dr


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

    # do additional convolutions to reduce edge case possibilites
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
    conv = np.pad(conv, 1, 'edge')
    return conv


def alpha_blend(img_back, img_front, a_front, a_back=1.):
    """
    Creates a single rgb array for overlaying the two images with their
    relative alpha values
    """
    tmp = a_back * (1. - a_front)
    return (img_front * a_front + img_back * tmp) / (a_front + tmp)


def highlight(img, mask, color=None, alpha=0.5):
    """
    highlight the given mask in the image using alpha blend (edits
    img object)
    """
    color = color if color else [0., 1., 0.]
    img[mask] = alpha_blend(img[mask], np.array(color), alpha)
    return img


def anti_highlight(img, mask, alpha=0.5):
    """
    highlight the given mask by dimming the rest of the image (edits
    img object)
    """
    anti_mask = np.logical_not(mask)
    return highlight(img, anti_mask, color=[0., 0., 0.], alpha=alpha)



#%%


if __name__ == '__main__':
    TSAgui()
