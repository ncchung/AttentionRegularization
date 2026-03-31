import os
import csv
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import gc

# for pi0, qvalue and lfdr estimation
from statsmodels.stats.multitest import local_fdr
from qvalue.qvalue import qvalue, pi0est, plot_qvalue

from attnreg.attention import ViTWrapper, get_attention_map, get_class_map
from attnreg.bootstrap import BootstrapFunc, bootstrap_normal
from attnreg.plotting import plot_attention, plot_statistics, plot_statistics_in_ROI


    
def get_nulls(
    model: ViTWrapper, 
    image: torch.Tensor, 
    bootstrap_method: BootstrapFunc = bootstrap_normal,
    bootstrap_kwargs: dict | None = None,
    bootstrap_N: int = 1, 
    return_mean=True, 
    CDAM=False, 
    target_class=None,
) -> tuple[np.ndarray,list,torch.Tensor]:
    """
    Perturb a full image and get attention scores from the full pictures, as nulls
    """

    if bootstrap_kwargs is None:
        bootstrap_kwargs = {}

    nulls_cdam = []

    if image.ndim < 4:
        image = image.unsqueeze(0)

    B, C, H, W = image.shape

    perturbed_images = bootstrap_method(image, bootstrap_N, **bootstrap_kwargs) # [B, bootstrap_N, C, H, W]

    if image.device == 'cuda':
        null_am = get_attention_map(model, perturbed_images.flatten(0,1), return_raw=False, return_mean=return_mean) # [B*bootstrap_N,H,W]
    else:

        flatten_perturbed = perturbed_images.flatten(0,1)
        null_am = np.empty((B*bootstrap_N, H, W))

        for i in range(B*bootstrap_N):

            n_am = get_attention_map(model, flatten_perturbed[i].unsqueeze(0), return_raw=False, return_mean=return_mean)
            null_am[i] = n_am.squeeze(0)
        
    null_am_array = np.asarray(null_am.reshape(B, bootstrap_N, H, W), dtype=np.float32) #  [B, bootstrap_N, H, W]

    # THE CDAM PART HAS TO BE MODIFIED 
    if CDAM:

        for i in range(bootstrap_N):
        
            cdam = get_class_map(model, perturbed_images[i], target_class, return_raw=False, return_mean=return_mean, clip=False) # plug Activation and Grad too
            nulls_cdam.append(np.asarray(cdam.flatten(), dtype=np.float32))
            
        null_cdam = np.concatenate(nulls_cdam, axis=0)
    else:
        null_cdam = None

    gc.collect()
        
    return null_am_array, null_cdam, perturbed_images


def get_nulls_batch(
    model: ViTWrapper,
    batch: torch.Tensor,
    bootstrap_method,
    bootstrap_kwargs: dict | None = None,
    bootstrap_N: int = 1,
    return_mean: bool = True,
):
    """
    Batched null computation.

    Args:
        batch: torch.Tensor [B, C, H, W]

    Returns:
        null_am_array: np.ndarray [B, bootstrap_N, H, W]
    """

    if bootstrap_kwargs is None:
        bootstrap_kwargs = {}

    if batch.ndim != 4:
        raise ValueError("Batch must be [B, C, H, W]")

    B, C, H, W = batch.shape

    # [B, bootstrap_N, C, H, W]
    perturbed = bootstrap_method(batch, bootstrap_N, **bootstrap_kwargs)

    # Flatten → [B * bootstrap_N, C, H, W]
    flattened = perturbed.flatten(0, 1)

    # get_attention_map returns numpy array
    null_am = get_attention_map(
        model,
        flattened,
        return_raw=False,
        return_mean=return_mean,
    )  # shape: [B*bootstrap_N, H, W]

    # reshape back to [B, bootstrap_N, H, W]
    null_am_array = np.asarray(
        null_am.reshape(B, bootstrap_N, H, W),
        dtype=np.float32,
    )

    return null_am_array


def zstat(
    null: list[float], 
    obs: list[float] | None = None,
    reshape=True,
    draw=True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Calculate z-scores
    """
    
    null  =  np.asarray(null, dtype=np.float32)
    null_log = np.log(null.ravel())
    null_mean = np.mean(null_log)
    null_std = np.std(null_log)
    null_z = (null_log - null_mean) / null_std
    
    if draw:
        plt.figure()
        sns.histplot(null_z, bins=50, stat="probability", kde=False, color='grey').set_title("z-statistics of the bootstrap samples only")

    if reshape:
        null_z = null_z.reshape(null.shape)

    obs_z = None

    if obs is not None:
        
        obs =  np.asarray(obs, dtype=np.float32)
        obs_log = np.log(obs.ravel())
        obs_z = (obs_log - null_mean) / null_std
        
        if draw:
            sns.histplot(obs_z, bins=50, stat="probability", kde=False, color='green').set_title("z-statistics with mean and variance computed from the bootstrap")
            
        if reshape:
            obs_z = obs_z.reshape(obs.shape)
            
    return null_z, obs_z


def emp_p(stat, stat0, draw=False):
    original_shape = stat.shape

    stat = np.abs(stat.ravel())
    #stat = stat.ravel()
    stat0 = np.concatenate((stat, np.abs(stat0.ravel())))
    #stat0 = stat0.ravel()
    #stat0 = np.concatenate((stat, stat0.ravel()))
    m0 = len(stat0)

    stat0_sorted = np.sort(stat0)

    idx = np.searchsorted(stat0_sorted, stat, side='right')
    p = 1 - (idx / m0)

    return p.reshape(original_shape)

# DIFFERENT DEFINITION OF P-VALUES
def emp_p_plusone(stat, stat0, slow=False, draw=False):
    original_shape = stat.shape

    stat = np.abs(stat.ravel())
    stat0 = np.abs(stat0.ravel())
    m = len(stat)
    m0 = len(stat0)
    
    if slow:
        p = [(np.sum(stat0 > item) + 1.0)/(m0 + 1.0) for item in stat]
    else:
        stat0_sorted = np.sort(stat0)
        p = [1 - (np.searchsorted(stat0_sorted, item, side='right') / m0) for item in stat]

    if draw:
        plt.figure()
        sns.histplot(p, bins=20, kde=False, color='black').set_title("Empirical P-values using the Bootstrap")
        
    p = np.array(p).reshape(original_shape)
    return p


#from qvalue.qvalue import qvalue, pi0est, plot_qvalue
def lfdr(
    z: np.ndarray, 
    z0: np.ndarray | None = None,
    pi0_prior: float | str = "estimate",
    draw=False,
) -> tuple[np.ndarray, float]:
    """
    Compute local false discovery rate (LFDR)
    """
    
    if pi0_prior == "estimate" and z0 is None:
        ValueError("When pi0 is set to estimate, z0 must be provided")
        
    original_shape = z.shape
    z = z.flatten()
    if z0 is not None:
        if type(z0) is not list:
            z0 = z0.flatten()
            
    if pi0_prior == "estimate":
        p = emp_p(np.abs(z), np.abs(z0), draw=False)
        pi0_est = pi0est(np.array(p))
        pi0 = pi0_est.item() if hasattr(pi0_est, 'item') else pi0_est
        l = local_fdr(z, null_proportion=pi0)
    elif type(pi0_prior) == float:
        assert(pi0_prior >= 0 and pi0_prior <= 1), "pi0 must be between 0 and 1"
        l = local_fdr(z, null_proportion=pi0_prior)
        pi0 = pi0_prior
    else:
        l = local_fdr(z)
        pi0 = float(1)
        
    if draw:
        plt.figure()
        sns.histplot(l, bins=20, kde=False, color='black').set_title("Local FDR with pi0est="+str(round(pi0,3)))

    l = l.reshape(original_shape)
    return l, pi0


def attention_uncertainty(
    model: ViTWrapper, 
    image: torch.Tensor, 
    bootstrap_method: BootstrapFunc = bootstrap_normal, 
    bootstrap_kwargs: dict | None = None,
    bootstrap_N=1, 
    pi0_prior: float | str = "estimate", 
    return_mean=True, 
    draw=False, 
    CDAM=False, 
    target_class=None,
):
    """
    A high-level wrapper function to compute attention uncertainty
    """

    #original_shape = image.shape
    # image.requires_grad = True

    obs_am = get_attention_map(model, image, return_raw=False, return_mean=return_mean)[0] # we use it only with batch=1 Tensors

    if CDAM:
        
        obs_cdam = get_class_map(model, image, target_class, return_raw=False, return_mean=return_mean, clip=False) 
        
    else:
        obs_cdam = None
        
    null_am, null_cdam, perturbed_images = get_nulls(model, image, bootstrap_method=bootstrap_method, bootstrap_kwargs=bootstrap_kwargs, bootstrap_N=bootstrap_N, return_mean=return_mean, CDAM=CDAM)

    null_am_zz, obs_am_zz = zstat(null=null_am, obs=obs_am, draw=draw)
    p_am = emp_p(obs_am_zz, null_am_zz, draw=False)
    
    l, pi0 = lfdr(z=obs_am_zz, z0=null_am_zz, pi0_prior=pi0_prior, draw=draw)

    return l, obs_am_zz, null_am_zz, p_am, pi0, obs_am, null_am, perturbed_images, obs_cdam, null_cdam


def create_mask(path, img_size):
    """
    If path is a JPEG file -> returns one (img_size, img_size) bool numpy array.
    If path is a folder     -> returns a list of such arrays for all
                               images listed in metadata.csv.
    """

    def build_array(x, y, size=img_size, square_size=100):
        arr = np.zeros((size, size), dtype=bool)

        # Clip square to stay inside bounds
        x_end = min(x + square_size, size)
        y_end = min(y + square_size, size)

        arr[y:y_end, x:x_end] = True
        return arr

    # Determine folder
    if os.path.isfile(path):
        folder = os.path.dirname(path)
        single_file = True
        target_filename = os.path.basename(path)
    elif os.path.isdir(path):
        folder = path
        single_file = False
    else:
        raise ValueError("Provided path is neither a file nor a folder.")

    metadata_path = os.path.join(folder, "metadata.csv")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError("metadata.csv not found in the folder.")

    results = []

    with open(metadata_path, newline='') as csvfile:
        reader = csv.reader(csvfile)

        # Try to skip header automatically
        first_row = next(reader)
        try:
            float(first_row[1])
            rows = [first_row] + list(reader)
        except ValueError:
            rows = list(reader)

        for row in rows:
            filename, x, y = row[0], int(row[1]), int(row[2])

            if single_file:
                if filename == target_filename:
                    return build_array(x, y), x, y
            else:
                results.append(build_array(x, y))

    if single_file:
        raise ValueError(f"No entry found for file {target_filename} in metadata.csv")

    return results, x, y


def get_attention_and_bootstrap(
    model: ViTWrapper, 
    image: torch.Tensor | Image.Image, 
    bootstrap_method: BootstrapFunc = bootstrap_normal, 
    bootstrap_kwargs: dict | None = None,
    bootstrap_N=1,
    ROI: torch.Tensor | np.ndarray | None = None,
    return_mean=True,
):

    if isinstance(image, Image.Image):
        image = model.PIL_to_tensor(image)
        
    obs_am = get_attention_map(model, image, return_raw=False, return_mean=return_mean)[0]

    null_am, _, _ = get_nulls(model, image, bootstrap_method=bootstrap_method, bootstrap_kwargs=bootstrap_kwargs, bootstrap_N=bootstrap_N, return_mean=return_mean, CDAM=False)

    null_am = null_am[0]

    within_z_range = True

    if ROI is not None:

        if isinstance(ROI, torch.Tensor):
            ROI = ROI.detach().cpu().numpy()
        
        null_am_zz, obs_am_zz = zstat(null=null_am, obs=obs_am, draw=False)

        if abs(np.mean(obs_am_zz[ROI])) > 1:

            within_z_range = False
            

    return obs_am, null_am, within_z_range


def get_attention_and_bootstrap_batch(
    model: ViTWrapper, 
    batch: torch.Tensor, 
    bootstrap_method: BootstrapFunc = bootstrap_normal, 
    bootstrap_kwargs: dict | None = None,
    bootstrap_N=1,
    ROI: list[np.ndarray] | None = None,
    return_mean=True,
    max_batch_size=512,
):

    if bootstrap_kwargs is None:
        bootstrap_kwargs = {}
        
    image_B, _, image_H, image_W = batch.shape

    full_batch_size = image_B*(1+bootstrap_N)

    obs_am = np.empty((image_B, image_H, image_W))
    null_am = np.empty((image_B, bootstrap_N, image_H, image_W))

    if full_batch_size <= max_batch_size:

        perturbed_images = bootstrap_method(batch, bootstrap_N, **bootstrap_kwargs) # [B, bootstrap_N, C, H, W]

        full_batch = torch.cat( [batch, perturbed_images.flatten(0,1)], dim=0) # concatenate images and bootstrap into [B+B*bootstrap_N,C,H,W]

        # Process the full batch
        ams = get_attention_map(model, full_batch, return_raw=False, return_mean=return_mean) # [B+B*bootstrap_N,H,W]

        obs_am = ams[:image_B] # [B,H,W]
        null_am = ams[image_B:].reshape(image_B,bootstrap_N,image_H,image_W) # [B, bootstrap_N, H, W]

    else:

        for i, image in enumerate(batch):

            am = get_attention_map(model, image, return_raw=False, return_mean=return_mean) # [1, H, W]

            n_am, _, _ = get_nulls(model, image, bootstrap_method=bootstrap_method, bootstrap_kwargs=bootstrap_kwargs, bootstrap_N=bootstrap_N, return_mean=return_mean, CDAM=False) # [1, bootstrap_N, H, W]

            obs_am[i] = am.squeeze(0)
            null_am[i] = n_am.squeeze(0)


    within_z_range = [True] * image_B
        
    if ROI is not None:

        for i, roi in enumerate(ROI):

            null_am_zz, obs_am_zz = zstat(null=null_am[i].flatten(), obs=obs_am[i], draw=False)

            if abs(np.mean(obs_am_zz[roi])) > 1:

                within_z_range[i] = False
            
    return obs_am, null_am, within_z_range



### A high-level wrapper function to apply all regularization methods and return all objects
def regularize_attention_all(
    l: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    pi0: float,
    obs_am: np.ndarray,
    z_th=0.0,
    p_th=0.1,
    lfdr_th=0.1,
):
    """
    Regularize attention map using different methods and based on different statistics
    """

    z_mask = z >= z_th
    p_mask = p <= p_th
    l_mask = l <= lfdr_th

    # z-threshold regularization
    obs_zreg = obs_am.copy()
    obs_zreg = obs_am * z_mask
    
    # Use the z-regularized map as the base for further regularization
    obs_base = obs_zreg # obs_am

    # p-threshold regularization
    obs_p = obs_base.copy()
    obs_p = obs_zreg * p_mask

    # l shrinkage
    # obs_ppw = obs_base.copy()
    # obs_ppw = obs_am * (1-l)

    # l-threshold regularization
    obs_lfdr = obs_base.copy()
    obs_lfdr = obs_zreg * l_mask

    # pi0-threshold regularization
    obs_pi0 = obs_base.copy()
    pi0_th = np.percentile(p, 100 * (1 - pi0))
    obs_pi0[p > pi0_th] = 0
    
    # obs_ppwpi0 = obs_pi0.copy()
    # obs_ppwpi0 = obs_ppwpi0 * (1-l)
    
    return obs_p, None, obs_lfdr, obs_pi0, obs_zreg, l, z, p, pi0
    

def nonzeros_and_mean_percentiles_in_ROI(
    am: np.ndarray,
    rest_sorted: np.ndarray,
    roi_idx: np.ndarray,
):

    am_roi = am.ravel()[roi_idx]

    non_zeros_percentage = np.count_nonzero(am_roi) / len(am_roi) * 100
    
    # left = (
    #     np.searchsorted(rest_sorted, am_roi, side='left')
    #     / len(rest_sorted)
    # ) * 100

    left = np.searchsorted(rest_sorted, am_roi, side='left')
    #right = np.searchsorted(rest_sorted, am_roi, side='right')

    #percentiles_roi = (left + right) / 2 / len(rest_sorted) * 100
    percentiles_roi = left * 100 / len(rest_sorted)
    
    mean_percentile_roi = np.mean(percentiles_roi)

    return non_zeros_percentage, mean_percentile_roi
    

def regularize_from_attention_scores_ROI(
    obs_am: np.ndarray,
    null_am: np.ndarray,
    ROI: np.ndarray,
    p_th: float = 0.3,
    l_th: float = 0.3,
    z_th: float = 0.0,
    pi0_prior: float | str = "estimate", 
    draw_statistics=False,
    statistics_figsize=(18, 5),
    savename: str | None = None,
    zreg_base=False, # zreg_base = True means that "before regularization" corresponds to the attention scores after z-regularization
):

    roi_idx = np.flatnonzero(ROI)
    rest_idx = np.flatnonzero(~ROI)
    
    z0, z = zstat(null=null_am, obs=obs_am, draw=False)
    p = emp_p(z, z0, draw=False)
    
    l, pi0 = lfdr(z=z, z0=z0, pi0_prior=pi0_prior, draw=False)

    if draw_statistics:

        l_roi = l.ravel()[roi_idx]
        p_roi = p.ravel()[roi_idx]
        z_roi = z.ravel()[roi_idx]

        
        plot_statistics_in_ROI(z,z_roi,p_roi[z_roi > 0],l_roi[z_roi > 0],z0, savename=savename)
        
    obs_p, obs_ppw, obs_lfdr, obs_pi0, obs_zreg, l, z, p, pi0 = regularize_attention_all(l,z,p,pi0,obs_am,z_th=z_th,p_th=p_th,lfdr_th=l_th)

    if zreg_base is True:
        obs_base = obs_zreg
    else:
        obs_base = obs_am


    rest_sorted = np.sort(obs_base.ravel()[rest_idx])

    nz_orig, mean_perc_orig = nonzeros_and_mean_percentiles_in_ROI(obs_base,rest_sorted,roi_idx)
    nz_p, mean_perc_p = nonzeros_and_mean_percentiles_in_ROI(obs_p,rest_sorted,roi_idx)
    nz_lfdr, mean_perc_lfdr = nonzeros_and_mean_percentiles_in_ROI(obs_lfdr,rest_sorted,roi_idx)
    nz_pi0, mean_perc_pi0 = nonzeros_and_mean_percentiles_in_ROI(obs_pi0,rest_sorted,roi_idx)

    return obs_am, obs_zreg, obs_p, obs_lfdr, obs_pi0, mean_perc_orig, mean_perc_p, mean_perc_lfdr, mean_perc_pi0, nz_orig, nz_p, nz_lfdr, nz_pi0



def regularize(
    model_wrapper: ViTWrapper, 
    img_path: str, 
    bootstrap_function: BootstrapFunc = bootstrap_normal, 
    bootstrap_kwargs: dict | None = None,
    bootstrap_N: int = 1,
    p_th: float = 0.3, 
    l_th: float = 0.3, 
    z_th: float = 0.0, 
    p_percentile: int | None = None,
    l_percentile: int | None = None,
    pi0_prior: float | str = "estimate",
    draw_original=False,
    draw_intermediate=False,
    draw_statistics=False,
    statistics_figsize=(18, 5),
    draw_attention=False,
    attention_figsize=(20, 7),
    savename: str | None = None,
):
    """
    A high-level wrapper function to get an attention map for an image and regularize it using a given bootstrap method and different regularization thresholds
    """

    # Load image 
    sample_img, original_img = model_wrapper.load_img(img_path,show=draw_original)

    # Create a bootstrap sample and compute all the statistics for the image
    l, z, null_am_zz, p, pi0, obs_am, null_am, perturbed_images, _, _ = (
        attention_uncertainty(
            model_wrapper, 
            sample_img, 
            bootstrap_method=bootstrap_function, 
            bootstrap_kwargs=bootstrap_kwargs, 
            bootstrap_N=bootstrap_N, 
            pi0_prior=pi0_prior, 
            draw=draw_intermediate,
        )
    )

    if p_percentile:
        p_th = round(np.percentile(p, p_percentile),2)
    if l_percentile:
        l_th = round(np.percentile(l, l_percentile),2)

    obs_p, obs_ppw, obs_lfdr, obs_pi0, obs_zreg, l, z, p, pi0 = regularize_attention_all(l,z,p,pi0,obs_am,p_th=p_th,lfdr_th=l_th)

    if draw_statistics:
        plot_statistics(z,p,l,null_am_zz, savename=savename, figsize=statistics_figsize)

    if draw_attention:
        plot_attention(original=original_img, maps= [obs_am, obs_p, obs_lfdr, obs_pi0], stats = [],
           titles=["Attention", f"p < {p_th}", f"l < {l_th}", r"$\pi_0$ reg"], savename=savename, figsize=attention_figsize)
    
    return original_img, obs_am, obs_p, obs_lfdr, obs_pi0, l, z, p, pi0

