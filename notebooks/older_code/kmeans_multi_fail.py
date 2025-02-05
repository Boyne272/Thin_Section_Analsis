import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from tools import set_seed, get_img, progress_bar


class img_base():
    """
    The base needed to convert a image array into a 5d vector
    """
    
    def __init__(self, img, **kwargs):
        "setup the img and vectors"
        
        assert img.ndim == 3, "image must be a 3d array"
        assert img.shape[-1] == 3, "image must be of form (x ,y, rgb)"
        
        self.dim_y, self.dim_x = img.shape[:2]
        self.Np = self.dim_x * self.dim_y      # the number of pixels
        
        self.vectors = self.img_to_vectors(img)
        self.img = img
        
        # if a second image given
        if 'polar' in kwargs.keys():
            self.img_polar = kwargs['polar']
            assert self.img_polar.shape == img.shape, "images must have same dimensions"
            
            vecs_polar = self.img_to_vectors(self.img_polar)
            
             # keep the cordinates and both color values
            self.vectors = torch.cat([self.vectors, vecs_polar], dim=1)
        
    
    def img_to_vectors(self, img):
        """
        Convert 3d image array (x, y, rgb) into an array of 5d vectors
        (x, y, r, g, b)"
        """
        
        # the x and y cordinates
        X, Y = np.meshgrid(range(self.dim_x),
                           range(self.dim_y))
        
        # create the 5d vectors
        vecs = np.zeros([self.Np, 5])
        vecs[:, 0] = X.ravel()
        vecs[:, 1] = Y.ravel()
        vecs[:, 2] = img[:, :, 0].ravel()
        vecs[:, 3] = img[:, :, 1].ravel()
        vecs[:, 4] = img[:, :, 2].ravel()
        
        return torch.from_numpy(vecs).float()
    
    
    def tensor_to_mask(self, tensor):
        "Reshape a tensor of values into the original image shape"
        return tensor.view(self.dim_y, self.dim_x).cpu().numpy()
    
    
    def mask_aplha_values(self, mask, color='r'):
        "Take a 2d mask and return a 3d rgba mask for imshow overlaying"
        
        assert type(mask) == type(np.array([])), "mask must be a numpy array"
        assert mask.ndim == 2, "mask must be a 2d array"
        
        zeros = np.zeros_like(mask)
        
        rgba = np.dstack([zeros, zeros, zeros, mask])
        
        if color == 'r':
            rgba[:, :, 0] = mask
        elif color == 'g':
            rgba[:, :, 1] = mask
        elif color == 'b':
            rgba[:, :, 2] = mask
            
        return rgba
    
    
    def outline(self, mask):
        """
        Take a 2d mask and use a laplacian convolution to find the segment 
        outlines
        """
        
        assert type(mask) == type(np.array([])), "mask must be a numpy array"
        assert mask.ndim == 2, "mask must be a 2d array"
        
        laplacian = np.ones([3, 3])
        laplacian[1, 1] = -8
        edges = sig.convolve2d(mask, laplacian, mode='valid')
        edges = (edges > 0).astype(float)
        
        return edges
    

            
class bin_base():
    """
    The base needed to bin vectors into a given grid
    """
    
    def __init__(self, bin_grid, **kwargs):
        "Setup the image grid with binned pixels into"
        
        assert len(bin_grid) == 2, "bin grid must be 2d"
        assert type(bin_grid[0]) == int, "grid must be integers"
        
        self.Nx, self.Ny = bin_grid # number of bin divisions in x and y
        self.Nk = self.Nx * self.Ny # number of k means centroids
        
        self.bin_dx = self.dim_x / self.Nx  # size of bins in x
        self.bin_dy = self.dim_y / self.Ny  # size of bins in y
        
        assert self.bin_dx == int(self.bin_dx), \
            "Must be evenly divisible in the x axis"
        assert self.bin_dy == int(self.bin_dy), \
            "Must be evenly divisible in the y axis"
        
        self.adj_bins = self.find_adjasent_bins() # create adjasent bins list
        
        
    def bin_vectors(self, vecs):
        """
        Bin vectors with cordinates (x, y, . . .) into the grid
        """
        x_bins = (vecs[:, 0] / self.bin_dx).floor()
        y_bins = (vecs[:, 1] / self.bin_dy).floor()
        output = y_bins * self.Nx + x_bins
        return output.int()
    
    
    def find_adjasent_bins(self):
        """
        Create a list of which bins are adjasent to which bins
        """
        adj_bins = [] 
        for i in range(self.Nk):
            # find the cordinates of each bin in the bin grid
            x, y = self.index_to_cords(i)
            # find the neightbours of that cordinate
            cordinates = self.neighbours(x, y, self.Nx, self.Ny)
            # convert the cordinates back into an index
            indexs = [ self.cords_to_index(x_, y_) for x_,y_ in cordinates ]
            # store the indexs
            adj_bins.append(indexs)
        return adj_bins
    
    
    def neighbours(self, x, y, x_max, y_max, r=1):
        """
        Find the neighbours in radius r to the x and y cordinate
        (this includes itself as a neighbour)
        """
        return [(x_, y_) for x_ in range(x-r, x+r+1)
                         for y_ in range(y-r, y+r+1)
                         if (#(x != x_ or y != y_) and   # not the center
                             (0 <= x_ < x_max) and # not outside x range
                             (0 <= y_ < y_max))]   # not outside y range
    
    
    def index_to_cords (self, i):
        return i%self.Nx, int(i/self.Nx)
    
    def cords_to_index(self, x, y):
        return y * self.Nx + x
       
	   
class distance_metrics():
    """
    This holds all the different distance metric choices. Each function
    find the distance between every vector pair in two rank two tensors
    """
    
    def __init__(self, choice, **kwargs):
        
        if choice == 'normal':
            self.func = self.distance_normal
            
        elif choice == 'default':
            self.func = self.distance_scaled
            if 'factor' in kwargs.keys():
                self.factor = kwargs['factor']
            else:
                self.factor = 1
            
        elif choice == 'custom':
            assert 'dist_func' in kwargs.keys(), "must pass a function"
            self.func = kwargs['dist_func']
        
        else:
            print("ditance metric not recognised")
            raise(ValueError)
            
        # if a second image is given
        if 'polar' in kwargs.keys():
            self.distance = self.mulit_dist_wrap
            
            # set the choice of how to combine the distances
            if 'combo_opt' in kwargs.keys():
                if kwargs['combo_opt'] == 'max':
                    self.cobmo_opt = lambda t:t.max(dim=0)
                elif kwargs['combo_opt'] == 'min':
                    self.cobmo_opt = lambda t:t.min(dim=0)
                elif kwargs['combo_opt'] == 'mean':
                    self.cobmo_opt = lambda t:t.mean(dim=0)
                elif kwargs['combo_opt'] == 'sum':
                    self.cobmo_opt = lambda t:t.sum(dim=0)
                else:
                    assert False, "combo opt not recoginsed"
            else:
                self.cobmo_opt = lambda t:t.mean(dim=0)
            
        else:
            self.distance = self.func
    
    
    def mulit_dist_wrap(self, vecs, clusts):
        "Utalises the other dist function now taking the max in two vectors"
        
        vecs_a = vecs[:, :5]
        vecs_b = vecs[:, [0,1,5,6,7]]
        
        dist_a = self.func(vecs_a, clusts)
        dist_b = self.func(vecs_b, clusts)
        
        combo = torch.stack([dist_a, dist_b])
        
        return self.cobmo_opt(combo)
    
    
    def distance_normal(self, vecs, clusts):
        """
        Find the distance between every vector and every cluster
        vecs and clusts are 2 rank tensors with samples on the first dimension
        returns a rank 2 tensor with samples on the first dimension and distance
        to each cluster on the second
        """
        disp = clusts - vecs[:, None]
        return disp.norm(dim=2)

    
    def distance_scaled(self, vecs, clusts):
        """
        Find the distance between every vector and every cluster
        vecs and clusts are 2 rank tensors with samples on the first dimension
        returns a rank 2 tensor with samples on the first dimension and distance
        to each cluster on the second
        """
        col_clust, col_vec = clusts[:, 2:], vecs[:, 2:]
        pos_clust, pos_vec = clusts[:, :2], vecs[:, :2]
        
        col_dist = (col_clust - col_vec[:, None]).norm(dim=2)
        pos_dist = (pos_clust - pos_vec[:, None]).norm(dim=2)

        return col_dist + pos_dist * self.factor / np.sqrt(self.Np / self.Nk)
    
    

class kmeans_local(img_base, bin_base, distance_metrics):
    """
    Implements Kmeans clustering on an image in 5d color position space
    using localised bins on a regular grid to enforce locality.
    """
    
    def __init__(self, img, bin_grid, dist_metric='default', **kwargs):
        """
        Parameters
        ----------
        
        img : 3d numpy array
            Image to be segmented by kmeans,
            expected in shape [x_dim, y_dim, rgb] with all values in the
            interval [0, 1], not 255.
            
        bin_grid : tuple
            Length of 2, gives the number of initial partitions in x and y
            respectivly, therefore the number of segments is the product of
            these. Both must be a factor of the x, y dimensions for the whole
            image. This intial segmentation also restrains the kmeans centers
            in space, forcing locality of segments and speeding up the
            algorithm.
            
        dist_metric : string, optional
            Choose the of calculating distance between two vectors:
            - 'normal' finds the normal distance without scaling either
                position or color space
            - 'default' finds the normal distance with scaling the position
                space by the bin widths. Can also pass the kwarg 'factor'
                with an additional scaling factor for the position space.
            - 'custom' pass a fucntion to be used in the kwarg 'dist_func'
            
        'polar' kwarg : 3d numpy array, optional
            Pass another image with the same dimensions as img which will
            also be used in the kmeans algorithm. The distance of pixels
            from each cluster center in both spaces will be found and the
            maximum taken as the ditance from each point to each center.
            
        'combo_opt' kwarg: string
            How to combine the distance metrics on each image. Options are:
            - 'min'
            - 'max'
            - 'mean'
            - 'sum'
        """
        
        # validate the kwargs
        valid_args = ['factor', 'polar', 'combo_opt']
        for key in kwargs.keys():
            assert key in valid_args, "kwarg " + key + " was not recognised"
        
        # setup the image, bin_grid and distance metric base classes
        img_base.__init__(self, img, **kwargs)
        bin_base.__init__(self, bin_grid, **kwargs)
        distance_metrics.__init__(self, dist_metric, **kwargs)
        
        # sort each vector into a bin (fixed)
        self.vec_bins_tensor = self.bin_vectors(self.vectors)
        # store in a list with element i being the vectors in bin i
        self.vec_bins_list = [(self.vec_bins_tensor==i).nonzero().squeeze()
                              for i in range(self.Nk)]
        
        # make the initial clusters the same as the bins (changes)
        self.cluster_tensor = self.vec_bins_tensor.clone()
        # store the indexs of these tensors in a dictionary (changes)
        self.cluster_list = [(self.cluster_tensor==i).nonzero().squeeze()
                             for i in range(self.Nk)]
        
        # create the initial centroids
        self.centroids = torch.empty([self.Nk, 5], dtype=torch.float)
        self.update_centroids()
        
        
    def update_centroids(self):
        """
        Find the new center of each cluster as the mean of its constituent
        elemtens
        """
        for i in range(self.Nk):
            vecs_in_cluster = self.cluster_list[i]
            assert vecs_in_cluster.numel() > 0, 'no cluser should be empty'
            self.centroids[i] = self.vectors[vecs_in_cluster][:, :5].mean(dim=0)
            
            
    def update_clusters(self):
        """
        Find which vectors belong to which cluster by find the ditance
        between every vector in a bin and every centroid in that bin or
        the neighbouring bins.
        """
        
        # bin the centroids
        cent_bins_tensor = self.bin_vectors(self.centroids)
        
        # for every bin grid (same as number of centroids)
        for i in range(self.Nk):
            
            # find which centroids are in adjasent bins
            # Could be optimised ##################
            adjacent_bins = self.adj_bins[i]
            centroid_is_adjasent = np.isin(cent_bins_tensor, adjacent_bins)
            centroids_to_search = np.where(centroid_is_adjasent)[0]
            centroids_to_search = torch.from_numpy(centroids_to_search).int()
            
            # find the distance between every vector and each centroid
            vecs_in_bin = self.vec_bins_list[i]
            vecs_tensors = self.vectors[vecs_in_bin]
            cents_tensors = self.centroids[centroids_to_search.long()]
            dist = self.distance(vecs_tensors, cents_tensors) 
            
            # find which centroids are the closest and update them
            min_indexs = torch.argmin(dist, dim=1).long()
            min_clusters = centroids_to_search[min_indexs]
            self.cluster_tensor[vecs_in_bin] = min_clusters
        
        # recacluate the cluster dictionary
        for i in range(self.Nk):
            self.cluster_list[i] = (self.cluster_tensor==i).nonzero().squeeze()     
    
    
    def iterate(self, n_iter):
        "loop for n_iter iterations with progress bar"
        
        # create the progress bar
        self.progress_bar = progress_bar(n_iter)
    
        for i in range(n_iter):
            self.update_clusters()
            self.update_centroids()
            self.progress_bar(i)
        
        
    def plot(self, option='default', ax=None, path=None):
        """
        Plot a one fo the following options on an axis if specified:
            - 'default' orinal image and the segement outlines
            - 'edges' just the outlines in a transparent manner
            - 'img' just the orinal image
            - 'centers' each kmean centroid
            - 'bin_edges' the bin mesh used
            - 'bins' both 'img' and 'bin_edges'
            - 'time' the iterations vs time if iterate was called
            - 'polar' the polarised image and segmentation if it was given
            - 'both' does both 'polar' and 'default' on a multiplot
                (ignores given axis)
        
        If path is given the image will be saved on that path.
        """
        
        # validate opiton
        valid_ops = ['default', 'edges', 'img', 'centers', 'bins', 'time',
                     'polar', 'both', 'bin_edges']
        assert option in valid_ops, "option not recoginsed"
        
        # create an axis if not given
        if ax == None:
            fig, ax = plt.subplots(figsize=[22, 22])
            
        if option == 'default':
            self.plot('img', ax)
            self.plot('edges', ax)
            ax.set(title='Image Segmentation (default)')
            
        elif option == 'edges':
            mask = self.tensor_to_mask(self.cluster_tensor)
            mask = self.outline(mask)
            mask = self.mask_aplha_values(mask)
            ax.imshow(mask)
            ax.set(title='Image Segmentation (default)')
            
        elif option == 'img':
            ax.imshow(self.img)
            ax.set(title='Image Segmentation (img)')
            
        elif option == 'centers':
            ax.plot(self.centroids[:, 0].cpu().numpy(),
                    self.centroids[:, 1].cpu().numpy(), 'm*', ms=20)
            ax.set(title='Image Segmentation (centers)')

        elif option == 'bin_edges':
            mask = self.tensor_to_mask(self.vec_bins_tensor)
            mask = self.outline(mask)
            mask = self.mask_aplha_values(mask)
            ax.imshow(mask)
            ax.set(title='Image Segmentation (bins)')
            
        elif option == 'bins':
            mask = self.tensor_to_mask(self.vec_bins_tensor)
            ax.imshow(mask)
            ax.set(title='Image Segmentation (bins)')
            
        elif option == 'polar':
            assert hasattr(self, 'img_polar'), 'no polarised image was given'
            ax.imshow(self.img_polar)
            self.plot('edges', ax)
            ax.set(title='Image Segmentation (polar)')
            
        elif option == 'both':
            
            assert hasattr(self, 'img_polar'), 'no polarised image was given'
            try: plt.close(fig)
            except: pass
            
            fig, axs = plt.subplots(2,1, figsize=[44, 22])
            
            self.plot('polar', ax=axs[0])
            self.plot('default', ax=axs[1])
            
        elif option == 'time':
            assert hasattr(self, 'progress_bar'), 'must call iterate to use this'
            self.progress_bar.plot_time(ax)
            ax.set(title='Image Segmentation (time)')
            
        # save the figure if wanted
        if path:
            plt.savefig(path)
            
            
#     def save_segmentation(self, path):
#         "Save the segmentation mask for future use"
            
    
            
if __name__ == '__main__':
    # run an example

    # setup
    set_seed(10)
    img = get_img("images/TX1_white_cropped.tif")
    img_polar = get_img("images/TX1_polarised_cropped.tif")
    obj_both = kmeans_local(img, [20,15], polar=img_polar, combo_opt='sum')

    obj_both.iterate(10)
    obj_both.plot('both')
    plt.gca().set(title='Combined after 10 Iterations')