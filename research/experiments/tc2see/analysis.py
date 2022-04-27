import numpy as np
import re
from pathlib import Path
import nilearn as ni
from nilearn import image, plotting
import matplotlib.pyplot as plt


def cosine_distance(x, y, axis=0):
    """ calculate the cosine distance for multidimensional arrays. """
    x = np.asarray(x)
    y = np.asarray(y)
    scalar_product = np.sum(x * y, axis=axis)
    lengthX = np.linalg.norm(x, axis=axis)
    lengthY = np.linalg.norm(y, axis=axis)
    return 1 - scalar_product / (lengthX * lengthY)

def pearson(x, y, axis=1):
    mu_x = np.mean(x, axis=axis)
    mu_y = np.mean(y, axis=axis)
    std_x = np.std(x, axis=axis)
    std_y = np.std(y, axis=axis)
    return np.mean(((x-mu_x[:, None])[:, None]*(y-mu_y[:, None])[None, :]), axis=-1) / (std_x[:, None] * std_y[None, :])

def get_rsm(activations, type="cosine"):
    if type == "cosine":
        return cosine_distance(activations[:, None, :], activations[None, :, :], axis=2)
    elif type == "pearson":
        return pearson(activations, activations)
    else:
        raise ValueError(type)

def get_design_matrix(frame_times, events, keep_order=False, hrf_model='glover'):
    from nilearn.glm.first_level import make_first_level_design_matrix

    index = 0
    def category(path):
        nonlocal index
        if path == "+":
            return "fixation_cross"
        if keep_order is True:
            index += 1
            return f"_{index:03d}" + re.match(r"\d*\.(.*)", Path(path).stem).groups()[0]  # .replace(".", "_").replace("_", "")[4:]
        return Path(path).stem.split('.')[1]  # .replace(".", "_").replace("_", "")[4:]

    events["trial_type"] = [category(s) for s in events.stimulus]
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events[["onset", "duration", "trial_type"]],
        hrf_model=hrf_model,
    )
    #plotting.plot_design_matrix(design_matrix)
    # plotting.show()

    vectors = []
    names = []
    for i, name in enumerate(design_matrix.columns):
        if name in ["fixation_cross", "constant"] or name.startswith("drift"):
            continue
        zeros = np.zeros(design_matrix.shape[1])
        zeros[i] = 1
        vectors.append(zeros)
        names.append(name)

    return design_matrix

def fit_glm(fmri_img, design_matrix, mask):
    from nilearn.glm.first_level import FirstLevelModel
    fmri_glm = FirstLevelModel(minimize_memory=False, mask_img=mask)
    fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrix)
    return fmri_glm

visual = None
def get_glm_activations(d, fmri_glm, vectors, mask_type="gm", r=None):
    global visual

    activations = []
    if r is not None:
        r2_mask = ni.image.math_img(f"a > {r}", a=fmri_glm.r_square[0]).slicer[:, :, :, 0]
    for vec in vectors:
        activations_1 = fmri_glm.compute_contrast(vec)

        if mask_type == "gm":
            gm_mask = ni.image.math_img("a >= 0.5", a=ni.image.resampling.resample_to_img(d.gm_probseg, d.fmri_img))
        elif mask_type == "visual":
            if visual is None:
                dataset_ju = ni.datasets.fetch_atlas_juelich('maxprob-thr0-1mm')
                visual = ni.image.math_img("(a >= 48)*(a < 48+5)", a=dataset_ju.maps)
            gm_mask = ni.image.resample_to_img(visual, d.fmri_img, interpolation="nearest")
        else:
            raise NameError(mask_type)
        if r is not None:
            gm_mask = ni.image.math_img("a * b", a=gm_mask, b=r2_mask)
        act = ni.masking.apply_mask(activations_1, gm_mask)
        activations.append(act)

    activations = np.array(activations)
    return activations

def plot_rsm(X, names, rescale=True, ax=None, output_file=None):
    """Plot a design matrix provided as a :class:`pandas.DataFrame`.

    Parameters
    ----------
    design matrix : :class:`pandas.DataFrame`
        Describes a design matrix.

    rescale : :obj:`bool`, optional
        Rescale columns magnitude for visualization or not.
        Default=True.

    ax : :class:`matplotlib.axes.Axes`, optional
        Handle to axes onto which we will draw the design matrix.
    %(output_file)s

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The axes used for plotting.

    """
    # normalize the values per column for better visualization
    #_, X, names = check_design_matrix(design_matrix)
    #if rescale:
    #    X = X / np.maximum(1.e-12, np.sqrt(
    #        np.sum(X ** 2, 0)))  # pylint: disable=no-member
    if ax is None:
        max_len = np.max([len(str(name)) for name in names])
        fig_height = 1 + .1 * X.shape[0] + .04 * max_len
        if fig_height < 3:
            fig_height = 3
        elif fig_height > 10:
            fig_height = 10
        plt.figure(figsize=(1 + .23 * len(names), fig_height))
        ax = plt.subplot(1, 1, 1)

    im = ax.imshow(X, interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar(im)
    #ax.set_label('conditions')
    #ax.set_ylabel('scan number')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha='left')
    # Set ticks above, to have a display more similar to the display of a
    # corresponding dataframe
    ax.xaxis.tick_top()

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
        ax = None
    return ax
