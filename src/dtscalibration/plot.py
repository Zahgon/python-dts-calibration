import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


def plot_residuals_reference_sections(
    resid,
    sections,
    fig=None,
    title=None,
    plot_avg_std=None,
    plot_names=True,
    robust=True,
    units="",
    fig_kwargs=None,
    method="split",
    cmap="RdBu_r",
):
    """Analyze the residuals of the reference sections, between the Stokes
    signal and a best-fit
    decaying exponential.

    Parameters
    ----------
    plot_avg_std
    resid : DataArray
        The residuals of the fit to estimate the noise in the measured
        Stokes signal. is returned by `variance_stokes_*()`
    sections : Dict[str, List[slice]]
        The sections obj is normally used to set DataStore.sections, now is
        used toobtain the
        section names to plot the names on top of the residuals.
    fig : Figurehandle, optional
    title : str, optional
        Adds a title to the plot
    plot_names : bool
        Whether the names of the sections are plotted on top of the residuals
    method: str
        'split' will remove the distance between sections to cut down on the
        whitespace.
        'single' will use the previous method, where all sections are in one
        plot.
    cmap : str
        Matplotlib colormap to use for the residual plot. By default it will
        use a diverging colormap.

    Returns:
    --------
    fig : Figurehandle

    """
    pass


def plot_residuals_reference_sections_single(
    resid,
    fig=None,
    title=None,
    plot_avg_std=None,
    plot_names=True,
    sections=None,
    robust=True,
    units="",
    fig_kwargs=None,
):
    """Analyze the residuals of the reference sections, between the Stokes
    signal and a best-fit
    decaying exponential.

    Parameters
    ----------
    plot_avg_std
    resid : DataArray
        The residuals of the fit to estimate the noise in the measured
        Stokes signal. is returned by `variance_stokes_*()`
    fig : Figurehandle, optional
    title : str, optional
        Adds a title to the plot
    plot_names : bool
        Whether the names of the sections are plotted on top of the residuals
    sections : Dict[str, List[slice]]
        The sections obj is normally used to set DataStore.sections, now is
        used toobtain the
        section names to plot the names on top of the residuals.
    'time' : str
        Name of the time dimension to average/take the variance of
    "x" : str
        Name of the spatial dimension

    Returns:
    --------
    fig : Figurehandle

    """
    pass


def plot_accuracy(
    accuracy,
    accuracy_x_avg,
    accuracy_time_avg,
    precision_x_avg=None,
    precision_time_avg=None,
    real_accuracy_time_avg=None,
    fig=None,
    title=None,
    plot_names=True,
    sections=None,
):
    """Analyze the residuals of the reference sections, between the Stokes
    signal and a best-fit
    decaying exponential.

    Parameters
    ----------
    plot_avg_std
    resid : DataArray
        The residuals of the fit to estimate the noise in the measured
        Stokes signal. is returned by `variance_stokes_*()`
    fig : Figurehandle, optional
    title : str, optional
        Adds a title to the plot
    plot_names : bool, optional
        Whether the names of the sections are plotted on top of the residuals
    sections : Dict[str, List[slice]]
        The sections obj is normally used to set DataStore.sections, now is
        used toobtain the
        section names to plot the names on top of the residuals.
    "x" : str
        Name of the spatial dimension

    Returns:
    --------
    fig : Figurehandle

    """
    pass


def plot_sigma_report(
    ds, sections, temp_label, temp_var_acc_label, temp_var_prec_label=None, itimes=None
):
    """Returns two sub-plots. first a temperature with confidence boundaries.

    Parameters
    ----------
    ds
    sections
    temp_label
    temp_var_label
    itimes
    """
    pass


def plot_location_residuals_double_ended(
    ds, werr, hix, tix, ix_sec, ix_match_not_cal, nt
):
    pass
