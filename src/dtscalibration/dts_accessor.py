"""The xarray accessor for DTS calibration."""

import dask.array as da
import numpy as np
import scipy.stats as sst
import xarray as xr
import yaml

from dtscalibration.calibrate_utils import calibrate_double_ended_helper
from dtscalibration.calibrate_utils import calibration_single_ended_helper
from dtscalibration.calibrate_utils import parse_st_var
from dtscalibration.calibration.section_utils import set_matching_sections
from dtscalibration.calibration.section_utils import set_sections
from dtscalibration.calibration.section_utils import validate_no_overlapping_sections
from dtscalibration.calibration.section_utils import validate_sections
from dtscalibration.calibration.section_utils import validate_sections_definition
from dtscalibration.dts_accessor_utils import ParameterIndexDoubleEnded
from dtscalibration.dts_accessor_utils import ParameterIndexSingleEnded
from dtscalibration.dts_accessor_utils import get_params_from_pval_double_ended
from dtscalibration.dts_accessor_utils import get_params_from_pval_single_ended
from dtscalibration.dts_accessor_utils import ufunc_per_section_helper
from dtscalibration.io.utils import dim_attrs


@xr.register_dataset_accessor("dts")
class DtsAccessor:
    """An xarray accessor for DTS calibration."""

    def __init__(self, xarray_obj):
        """Initialize the DtsAccessor with an xarray object.

        Parameters
        ----------
        xarray_obj : xarray.Dataset
            The xarray object to be used for DTS calibration.
        """
        # cache xarray_obj
        self._obj = xarray_obj
        self.attrs = xarray_obj.attrs

        # alias commonly used variables
        self.x = xarray_obj.x
        self.nx = self.x.size
        self.time = xarray_obj.time
        self.nt = self.time.size

        # None if doesn't exist
        self.st = xarray_obj.get("st")
        self.ast = xarray_obj.get("ast")
        self.rst = xarray_obj.get("rst")
        self.rast = xarray_obj.get("rast")

        self.acquisitiontime_fw = xarray_obj.get("userAcquisitionTimeFW")
        self.acquisitiontime_bw = xarray_obj.get("userAcquisitionTimeBW")

    def __repr__(self):
        """Return a string representation of the DtsAccessor object."""
        # __repr__ from xarray is used and edited.
        #   'xarray' is prepended. so we remove it and add 'dtscalibration'
        s = xr.core.formatting.dataset_repr(self._obj)
        name_module = type(self._obj).__name__
        preamble_new = f"<dtscalibration.{name_module}>"

        # Add sections to new preamble
        preamble_new += "\nSections:"
        if hasattr(self._obj, "_sections") and self.sections:
            preamble_new += "\n"
            unit = self.x.attrs.get("unit", "")

            for k, v in self.sections.items():
                preamble_new += f"    {k: <23}"

                # Compute statistics reference section timeseries
                sec_stat = f"({float(self._obj[k].mean()):6.2f}"
                sec_stat += f" +/-{float(self._obj[k].std()):5.2f}"
                sec_stat += "\N{DEGREE SIGN}C)\t"
                preamble_new += sec_stat

                # print sections
                vl = [f"{vi.start:.2f}{unit} - {vi.stop:.2f}{unit}" for vi in v]
                preamble_new += " and ".join(vl) + "\n"

        else:
            preamble_new += 18 * " " + "()\n"

        # add new preamble to the remainder of the former __repr__
        len_preamble_old = 8 + len(name_module) + 2

        # untill the attribute listing
        attr_index = s.find("Attributes:")

        # abbreviate attribute listing
        attr_list_all = s[attr_index:].split(sep="\n")
        if len(attr_list_all) > 10:
            s_too_many = ["\n.. and many more attributes. See: ds.attrs"]
            attr_list = attr_list_all[:10] + s_too_many
        else:
            attr_list = attr_list_all

        s_out = preamble_new + s[len_preamble_old:attr_index] + "\n".join(attr_list)

        # return new __repr__
        return s_out

    # noinspection PyIncorrectDocstring
    @property
    def sections(self):
        """Define calibration sections.

        Each section requires a reference temperature time series, such as the temperature measured by an
        external temperature sensor. They should already be part of the DataStore object.

        Please look at the example notebook on `sections` if you encounter difficulties.

        Parameters
        ----------
        sections : Dict[str, List[slice]]
            Sections are defined in a dictionary with its keywords of the names of the reference
            temperature time series. Its values are lists of slice objects, where each slice object
            is a stretch.

        Returns:
        --------
        dict
            A dictionary with the names of the reference temperature time series as keys and lists of slice objects as values.
        """
        pass

    @sections.deleter
    def sections(self):
        pass

    @sections.setter
    def sections(self, value):
        pass

    # noinspection PyIncorrectDocstring
    @property
    def matching_sections(self):
        """Define calibration sections.

        Each matching_section requires a reference temperature time series, such as the temperature measured by an
        external temperature sensor. They should already be part of the DataStore object.

        Please look at the example notebook on `matching_sections` if you encounter difficulties.

        Parameters
        ----------
        matching_sections : List[Tuple[slice, slice, bool]], optional
            Provide a list of tuples. A tuple per matching section. Each tuple has three items. The first two items are
            the slices of the sections that are matched. The third item is a boolean and is True if the two sections have
            a reverse direction ("J-configuration").

        Returns:
        --------
        dict
            A dictionary with the names of the reference temperature time series as keys and lists of slice objects as values.
        """
        pass

    @matching_sections.deleter
    def matching_sections(self):
        pass

    @matching_sections.setter
    def matching_sections(self, value):
        pass

    def get_default_encoding(self, time_chunks_from_key=None):
        """Returns a dictionary with sensible compression setting for writing netCDF files.

        Returns:
        --------
        dict
            A dictionary with sensible compression setting for writing netCDF files.

        """
        pass

    def get_timeseries_keys(self):
        """Returns a list of the keys of the time series variables."""
        pass

    def ufunc_per_section(
        self,
        sections=None,
        func=None,
        label=None,
        subtract_from_label=None,
        temp_err=False,
        x_indices=False,
        ref_temp_broadcasted=False,
        calc_per="stretch",
        suppress_section_validation=False,
        **func_kwargs,
    ):
        """Functions applied to parts of the cable.
        
        Super useful,many options and slightlycomplicated.

        The function `func` is taken over all the timesteps and calculated
        per `calc_per`. This is returned as a dictionary

        Parameters
        ----------

        sections : Dict[str, List[slice]], optional
            If `None` is supplied, `ds.sections` is used. Define calibration
            sections. Each section requires a reference temperature time series,
            such as the temperature measured by an external temperature sensor.
            They should already be part of the DataStore object. `sections`
            is defined with a dictionary with its keywords of the
            names of the reference temperature time series. Its values are
            lists of slice objects, where each slice object is a fiber stretch
            that has the reference temperature. Afterwards, `sections` is stored
            under `ds.sections`.
        func : callable, str
            A numpy function, or lambda function to apple to each 'calc_per'.
        label
        subtract_from_label
        temp_err : bool
            The argument of the function is label minus the reference
            temperature.
        x_indices : bool
            To retreive an integer array with the indices of the
            x-coordinates in the section/stretch. The indices are sorted.
        ref_temp_broadcasted : bool
        calc_per : {'all', 'section', 'stretch'}
        func_kwargs : dict
            Dictionary with options that are passed to func

        TODO: Spend time on creating a slice instead of appendng everything\
        to a list and concatenating after.

        Returns:
        --------
        dict
            A dictionary with the keys of the sections and the values are the
            result of the function applied to the data in the section.

        Examples:
        ---------
        1. Calculate the variance of the residuals in the along ALL the\
        reference sections wrt the temperature of the water baths

        >>> tmpf_var = d.ufunc_per_section(
        >>>     sections=sections, 
        >>>     func='var',
        >>>     calc_per='all',
        >>>     label='tmpf',
        >>>     temp_err=True)

        2. Calculate the variance of the residuals in the along PER\
        reference section wrt the temperature of the water baths

        >>> tmpf_var = d.ufunc_per_section(
        >>>     sections=sections, 
        >>>     func='var',
        >>>     calc_per='stretch',
        >>>     label='tmpf',
        >>>     temp_err=True)

        3. Calculate the variance of the residuals in the along PER\
        water bath wrt the temperature of the water baths

        >>> tmpf_var = d.ufunc_per_section(
        >>>     sections=sections, 
        >>>     func='var',
        >>>     calc_per='section',
        >>>     label='tmpf',
        >>>     temp_err=True)

        4. Obtain the coordinates of the measurements per section

        >>> locs = d.ufunc_per_section(
        >>>     sections=sections, 
        >>>     func=None,
        >>>     label='x',
        >>>     temp_err=False,
        >>>     ref_temp_broadcasted=False,
        >>>     calc_per='stretch')

        5. Number of observations per stretch

        >>> nlocs = d.ufunc_per_section(
        >>>     sections=sections, 
        >>>     func=len,
        >>>     label='x',
        >>>     temp_err=False,
        >>>     ref_temp_broadcasted=False,
        >>>     calc_per='stretch')

        6. broadcast the temperature of the reference sections to\
        stretch/section/all dimensions. The value of the reference\
        temperature (a timeseries) is broadcasted to the shape of self[\
        label]. The self[label] is not used for anything else.

        >>> temp_ref = d.ufunc_per_section(
        >>>     label='st',
        >>>     ref_temp_broadcasted=True,
        >>>     calc_per='all')

        7. x-coordinate index

        >>> ix_loc = d.ufunc_per_section(sections=sections, x_indices=True)

        Notes:
        ------
        If `self[label]` or `self[subtract_from_label]` is a Dask array, a Dask
        array is returned else a numpy array is returned
        """
        if not suppress_section_validation:
            validate_sections_definition(sections=sections)
            validate_no_overlapping_sections(sections=sections)

        if temp_err or ref_temp_broadcasted:
            for k in sections:
                assert (
                    k in self._obj
                ), f"{k} is not in the Dataset but is in `sections` and is required to compute temp_err"

        dataarray = None if label is None else self._obj[label]

        if x_indices:
            x_coords = self.x
            reference_dataset = None

        else:
            x_coords = None
            reference_dataset = {k: self._obj[k] for k in sections}

        out = ufunc_per_section_helper(
            x_coords=x_coords,
            sections=sections,
            func=func,
            dataarray=dataarray,
            subtract_from_dataarray=subtract_from_label,
            reference_dataset=reference_dataset,
            subtract_reference_from_dataarray=temp_err,
            ref_temp_broadcasted=ref_temp_broadcasted,
            calc_per=calc_per,
            **func_kwargs,
        )
        return out

    def calibrate_single_ended(
        self,
        sections,
        st_var,
        ast_var,
        method="wls",
        solver="sparse",
        p_val=None,
        p_var=None,
        p_cov=None,
        matching_sections=None,
        trans_att=[],
        fix_gamma=None,
        fix_dalpha=None,
        fix_alpha=None,
    ):
        r"""Calibrate the measured data of a single ended setup to temperature.
        
        Calibrate the Stokes (`ds.st`) and anti-Stokes (`ds.ast`) data to
        temperature using fiber sections with a known temperature
        (`ds.sections`) for single-ended setups. 
        
        The calibrated temperature is
        stored under `ds.tmpf` and its variance under `ds.tmpf_var`.

        In single-ended setups, Stokes and anti-Stokes intensity is measured
        from a single end of the fiber. The differential attenuation is assumed
        constant along the fiber so that the integrated differential attenuation
        may be written as (Hausner et al, 2011):

        .. math::

            \int_0^x{\Delta\alpha(x')\,\mathrm{d}x'} \approx \Delta \alpha x

        The temperature can now be written from Equation 10 [1]_ as:

        .. math::

            T(x,t)  \approx \frac{\gamma}{I(x,t) + C(t) + \Delta\alpha x}

        where

        .. math::

            I(x,t) = \ln{\left(\frac{P_+(x,t)}{P_-(x,t)}\right)}

        .. math::

            C(t) = \ln{\left(\frac{\eta_-(t)K_-/\lambda_-^4}{\eta_+(t)K_+/\lambda_+^4}\right)}

        where :math:`C` is the lumped effect of the difference in gain at
        :math:`x=0` between Stokes and anti-Stokes intensity measurements and
        the dependence of the scattering intensity on the wavelength. The
        parameters :math:`P_+` and :math:`P_-` are the Stokes and anti-Stokes
        intensity measurements, respectively.
        The parameters :math:`\gamma`, :math:`C(t)`, and :math:`\Delta\alpha`
        must be estimated from calibration to reference sections, as discussed
        in Section 5 [1]_. The parameter :math:`C` must be estimated
        for each time and is constant along the fiber. :math:`T` in the listed
        equations is in Kelvin, but is converted to Celsius after calibration.

        Parameters
        ----------

        p_val : array-like, optional
            Define `p_val`, `p_var`, `p_cov` if you used an external function
            for calibration. Has size 2 + `nt`. First value is :math:`\gamma`,
            second is :math:`\Delta \alpha`, others are :math:`C` for each
            timestep.
        p_var : array-like, optional
            Define `p_val`, `p_var`, `p_cov` if you used an external function
            for calibration. Has size 2 + `nt`. First value is :math:`\gamma`,
            second is :math:`\Delta \alpha`, others are :math:`C` for each
            timestep.
        p_cov : array-like, optional
            The covariances of `p_val`.
            If set to False, no uncertainty in the parameters is propagated
            into the confidence intervals. Similar to the spec sheets of the DTS
            manufacturers. And similar to passing an array filled with zeros.
        sections : Dict[str, List[slice]]
            Each section requires a reference temperature time series,
            such as the temperature measured by an external temperature sensor.
            They should already be part of the DataStore object. `sections`
            is defined with a dictionary with its keywords of the
            names of the reference temperature time series. Its values are
            lists of slice objects, where each slice object is a fiber stretch
            that has the reference temperature. Afterwards, `sections` is stored
            under `ds.sections`.
        st_var, ast_var : float, callable, array-like, optional
            The variance of the measurement noise of the Stokes signals in the
            forward direction. If `float` the variance of the noise from the
            Stokes detector is described with a single value.
            If `callable` the variance of the noise from the Stokes detector is
            a function of the intensity, as defined in the callable function.
            Or manually define a variance with a DataArray of the shape
            `ds.st.shape`, where the variance can be a function of time and/or
            x. Required if method is wls.
        method : {'wls',}
            Use `'wls'` for weighted least
            squares.
        solver : {'sparse', 'stats'}
            Either use the homemade weighted sparse solver or the weighted
            dense matrix solver of statsmodels. The sparse solver uses much less
            memory, is faster, and gives the same result as the statsmodels
            solver. The statsmodels solver is mostly used to check the sparse
            solver. `'stats'` is the default.
        matching_sections : List[Tuple[slice, slice, bool]], optional
            Provide a list of tuples. A tuple per matching section. Each tuple
            has three items. The first two items are the slices of the sections
            that are matched. The third item is a boolean and is True if the two
            sections have a reverse direction ("J-configuration").
        trans_att : iterable, optional
            Splices can cause jumps in differential attenuation. Normal single
            ended calibration assumes these are not present. An additional loss
            term is added in the 'shadow' of the splice. Each location
            introduces an additional nt parameters to solve for. Requiring
            either an additional calibration section or matching sections.
            If multiple locations are defined, the losses are added.
        fix_gamma : Tuple[float, float], optional
            A tuple containing two floats. The first float is the value of
            gamma, and the second item is the variance of the estimate of gamma.
            Covariances between gamma and other parameters are not accounted
            for.
        fix_dalpha : Tuple[float, float], optional
            A tuple containing two floats. The first float is the value of
            dalpha (:math:`\Delta \alpha` in [1]_), and the second item is the
            variance of the estimate of dalpha.
            Covariances between alpha and other parameters are not accounted
            for.
        fix_alpha : Tuple[array-like, array-like], optional
            A tuple containing two array-likes. The first array-like is the integrated
            differential attenuation of length x, and the second item is its variance.

        Returns:
        --------
        out : xarray.Dataset
            A Dataset with the calibrated temperature under `tmpf` and its
            variance under `tmpf_var`. The Dataset also contains the
            calibration parameters under `gamma`, `dalpha`, and `c`. The
            covariance matrix of the calibration parameters is stored.

        References:
        -----------
        .. [1] des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation
            of Temperature and Associated Uncertainty from Fiber-Optic Raman-
            Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235.
            https://doi.org/10.3390/s20082235

        Examples:
        ---------
        - `Example notebook 7: Calibrate single ended <https://github.com/\
    dtscalibration/python-dts-calibration/blob/main/examples/notebooks/\
    07Calibrate_single_wls.ipynb>`_

        """
        pass

    def calibrate_double_ended(
        self,
        sections,
        st_var,
        ast_var,
        rst_var,
        rast_var,
        method="wls",
        solver="sparse",
        p_val=None,
        p_var=None,
        p_cov=None,
        trans_att=[],
        fix_gamma=None,
        fix_alpha=None,
        matching_sections=None,
        verbose=False,
    ):
        r"""Calibrate the measured data of a double ended setup to temperature.
        
        See example notebook 8 for an explanation on how to use this function.
        Calibrate the Stokes (`ds.st`) and anti-Stokes (`ds.ast`) of the forward
        channel and from the backward channel (`ds.rst`, `ds.rast`) data to
        temperature using fiber sections with a known temperature
        (`ds.sections`) for double-ended setups. The calibrated temperature of
        the forward channel is stored under `ds.tmpf` and its variance under
        `ds.tmpf_var`, and that of the backward channel under `ds.tmpb` and
        `ds.tmpb_var`. The inverse-variance weighted average of the forward and
        backward channel is stored under `ds.tmpw` and `ds.tmpw_var`.
        In double-ended setups, Stokes and anti-Stokes intensity is measured in
        two directions from both ends of the fiber. The forward-channel
        measurements are denoted with subscript F, and the backward-channel
        measurements are denoted with subscript B. Both measurement channels
        start at a different end of the fiber and have opposite directions, and
        therefore have different spatial coordinates.
        The first processing step
        with double-ended measurements is to align the measurements of the two
        measurement channels so that they have the same spatial coordinates. The
        spatial coordinate :math:`x` (m) is defined here positive in the forward
        direction, starting at 0 where the fiber is connected to the forward
        channel of the DTS system; the length of the fiber is :math:`L`.
        Consequently, the backward-channel measurements are flipped and shifted
        to align with the forward-channel measurements. Alignment of the
        measurements of the two channels is prone to error because it requires
        the exact fiber length (McDaniel et al., 2018). Depending on the DTS system
        used, the forward channel and backward channel are measured one after
        another by making use of an optical switch, so that only a single
        detector is needed. However, it is assumed in this package that the
        forward channel and backward channel are measured simultaneously, so
        that the temperature of both measurements is the same. This assumption
        holds better for short acquisition times with respect to the timescale
        of the temperature variation, and when there is no systematic difference
        in temperature between the two channels. The temperature may be computed
        from the forward-channel measurements (Equation 10 [1]_) with:

        .. math::
            T_\mathrm{F} (x,t)  = \frac{\gamma}{I_\mathrm{F}(x,t) + \
    C_\mathrm{F}(t) + \int_0^x{\Delta\alpha(x')\,\mathrm{d}x'}}

        and from the backward-channel measurements with:

        .. math::
            T_\mathrm{B} (x,t)  = \frac{\gamma}{I_\mathrm{B}(x,t) + \
            
        C_\mathrm{B}(t) + \int_x^L{\Delta\alpha(x')\,\mathrm{d}x'}}
        with

        .. math::
            I(x,t) = \ln{\left(\frac{P_+(x,t)}{P_-(x,t)}\right)}

        .. math::
            C(t) = \ln{\left(\frac{\eta_-(t)K_-/\lambda_-^4}{\eta_+(t)K_+/\lambda_+^4}\right)}

        where :math:`C` is the lumped effect of the difference in gain at
        :param mc_conf_ints:
        :math:`x=0` between Stokes and anti-Stokes intensity measurements and
        the dependence of the scattering intensity on the wavelength. The
        parameters :math:`P_+` and :math:`P_-` are the Stokes and anti-Stokes
        intensity measurements, respectively.
        :math:`C_\mathrm{F}(t)` and :math:`C_\mathrm{B}(t)` are the
        parameter :math:`C(t)` for the forward-channel and backward-channel
        measurements, respectively. :math:`C_\mathrm{B}(t)` may be different
        from :math:`C_\mathrm{F}(t)` due to differences in gain, and difference
        in the attenuation between the detectors and the point the fiber end is
        connected to the DTS system (:math:`\eta_+` and :math:`\eta_-` in
        Equation~\ref{eqn:c}). :math:`T` in the listed
        equations is in Kelvin, but is converted to Celsius after calibration.
        The calibration procedure presented in van de
        Giesen et al. 2012 approximates :math:`C(t)` to be
        the same for the forward and backward-channel measurements, but this
        approximation is not made here.
        Parameter :math:`A(x)` (`ds.alpha`) is introduced to simplify the notation of the
        double-ended calibration procedure and represents the integrated
        differential attenuation between locations :math:`x_1` and :math:`x`
        along the fiber. Location :math:`x_1` is the first reference section
        location (the smallest x-value of all used reference sections).

        .. math::

            A(x) = \int_{x_1}^x{\Delta\alpha(x')\,\mathrm{d}x'}
        
        so that the expressions for temperature may be written as:
        
        .. math::
        
            T_\mathrm{F} (x,t) = \frac{\gamma}{I_\mathrm{F}(x,t) + D_\mathrm{F}(t) + A(x)},
            T_\mathrm{B} (x,t) = \frac{\gamma}{I_\mathrm{B}(x,t) + D_\mathrm{B}(t) - A(x)}
        
        where
        
        .. math::
            D_{\mathrm{F}}(t) = C_{\mathrm{F}}(t) + \int_0^{x_1}{\Delta\alpha(x')\,\mathrm{d}x'},
            D_{\mathrm{B}}(t) = C_{\mathrm{B}}(t) + \int_{x_1}^L{\Delta\alpha(x')\,\mathrm{d}x'}
        
        Parameters :math:`D_\mathrm{F}` (`ds.df`) and :math:`D_\mathrm{B}`
        (`ds.db`) must be estimated for each time and are constant along the fiber, and parameter
        :math:`A` must be estimated for each location and is constant over time.
        The calibration procedure is discussed in Section 6.
        :math:`T_\mathrm{F}` (`ds.tmpf`) and :math:`T_\mathrm{B}` (`ds.tmpb`)
        are separate
        approximations of the same temperature at the same time. The estimated
        :math:`T_\mathrm{F}` is more accurate near :math:`x=0` because that is
        where the signal is strongest. Similarly, the estimated
        :math:`T_\mathrm{B}` is more accurate near :math:`x=L`. A single best
        estimate of the temperature is obtained from the weighted average of
        :math:`T_\mathrm{F}` and :math:`T_\mathrm{B}` as discussed in
        Section 7.2 [1]_ .

        Parameters
        ----------
        p_val : array-like, optional
            Define `p_val`, `p_var`, `p_cov` if you used an external function
            for calibration. Has size `1 + 2 * nt + nx + 2 * nt * nta`.
            First value is :math:`\gamma`, then `nt` times
            :math:`D_\mathrm{F}`, then `nt` times
            :math:`D_\mathrm{B}`, then for each location :math:`D_\mathrm{B}`,
            then for each connector that introduces directional attenuation two
            parameters per time step.
        p_var : array-like, optional
            Define `p_val`, `p_var`, `p_cov` if you used an external function
            for calibration. Has size `1 + 2 * nt + nx + 2 * nt * nta`.
            Is the variance of `p_val`.
        p_cov : array-like, optional
            The covariances of `p_val`. Square matrix.
            If set to False, no uncertainty in the parameters is propagated
            into the confidence intervals. Similar to the spec sheets of the DTS
            manufacturers. And similar to passing an array filled with zeros.
        sections : Dict[str, List[slice]], optional
            If `None` is supplied, `ds.sections` is used. Define calibration
            sections. Each section requires a reference temperature time series,
            such as the temperature measured by an external temperature sensor.
            They should already be part of the DataStore object. `sections`
            is defined with a dictionary with its keywords of the
            names of the reference temperature time series. Its values are
            lists of slice objects, where each slice object is a fiber stretch
            that has the reference temperature. Afterwards, `sections` is stored
            under `ds.sections`.
        st_var, ast_var, rst_var, rast_var : float, callable, array-like, optional
            The variance of the measurement noise of the Stokes signals in the
            forward direction. If `float` the variance of the noise from the
            Stokes detector is described with a single value.
            If `callable` the variance of the noise from the Stokes detector is
            a function of the intensity, as defined in the callable function.
            Or manually define a variance with a DataArray of the shape
            `ds.st.shape`, where the variance can be a function of time and/or
            x. Required if method is wls.
        mc_sample_size : int, optional
            If set, the variance is also computed using Monte Carlo sampling.
            The number of Monte Carlo samples drawn used to estimate the
            variance of the forward and backward channel temperature estimates
            and estimate the inverse-variance weighted average temperature.
        conf_ints : iterable object of float
            A list with the confidence boundaries that are calculated. Valid
            values are between [0, 1].
        mc_da_random_state
            For testing purposes. Similar to random seed. The seed for dask.
            Makes random not so random. To produce reproducable results for
            testing environments.
        mc_remove_set_flag : bool
            Remove the monte carlo data set, from which the CI and the
            variance are calculated.
        variance_suffix : str, optional
            String appended for storing the variance. Only used when method
            is wls.
        method : {'wls', 'external'}
            Use `'wls'` for weighted least squares.
        solver : {'sparse', 'stats'}
            Either use the homemade weighted sparse solver or the weighted
            dense matrix solver of statsmodels. The sparse solver uses much less
            memory, is faster, and gives the same result as the statsmodels
            solver. The statsmodels solver is mostly used to check the sparse
            solver. `'stats'` is the default.
        trans_att : iterable, optional
            Splices can cause jumps in differential attenuation. Normal single
            ended calibration assumes these are not present. An additional loss
            term is added in the 'shadow' of the splice. Each location
            introduces an additional nt parameters to solve for. Requiring
            either an additional calibration section or matching sections.
            If multiple locations are defined, the losses are added.
        fix_gamma : Tuple[float, float], optional
            A tuple containing two floats. The first float is the value of
            gamma, and the second item is the variance of the estimate of gamma.
            Covariances between gamma and other parameters are not accounted
            for.
        fix_alpha : Tuple[array-like, array-like], optional
            A tuple containing two arrays. The first array contains the
            values of integrated differential att (:math:`A` in paper), and the
            second array contains the variance of the estimate of alpha.
            Covariances (in-) between alpha and other parameters are not
            accounted for.
        matching_sections : List[Tuple[slice, slice, bool]]
            Provide a list of tuples. A tuple per matching section. Each tuple
            has three items. The first two items are the slices of the sections
            that are matched. The third item is a boolean and is True if the two
            sections have a reverse direction ("J-configuration").
        matching_indices : array
            Provide an array of x-indices of size (npair, 2), where each pair
            has the same temperature. Used to improve the estimate of the
            integrated differential attenuation.
        verbose : bool
            Show additional calibration information

        Returns:
        --------
        out : xarray.Dataset
            A Dataset with the calibrated temperature under `tmpf` and its
            variance under `tmpf_var`. The calibrated temperature of the
            backward channel is stored under `tmpb` and `tmpb_var`. The
            inverse-variance weighted average of the forward and backward
            channel is stored under `tmpw` and `tmpw_var`. The Dataset also
            contains the calibration parameters under `gamma`, `df`, `db`,
            `alpha`, and `c`. The covariance matrix of the calibration
            parameters is stored.

        References:
        -----------
        .. [1] des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation
            of Temperature and Associated Uncertainty from Fiber-Optic Raman-
            Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235.
            https://doi.org/10.3390/s20082235

        Examples:
        ---------
        - `Example notebook 8: Calibrate double ended <https://github.com/
        dtscalibration/python-dts-calibration/blob/master/examples/notebooks/
        08Calibrate_double_wls.ipynb>`
        """
        pass

    def monte_carlo_single_ended(
        self,
        result,
        st_var,
        ast_var,
        conf_ints=[],
        mc_sample_size=100,
        da_random_state=None,
        reduce_memory_usage=False,
        mc_remove_set_flag=True,
    ):
        """The result object is what comes out of the single_ended_calibration routine).

        TODO: Use get_params_from_pval_single_ended() to extract parameter sets from mc
        """
        pass

    def monte_carlo_double_ended(
        self,
        result,
        st_var,
        ast_var,
        rst_var,
        rast_var,
        conf_ints,
        mc_sample_size=100,
        var_only_sections=False,
        exclude_parameter_uncertainty=False,
        da_random_state=None,
        mc_remove_set_flag=True,
        reduce_memory_usage=False,
    ):
        r"""Estimation of the confidence intervals for the temperatures measured with a double-ended setup.

        Double-ended setups require four additional steps to estimate the
        confidence intervals for the temperature. First, the variances of the
        Stokes and anti-Stokes intensity measurements of the forward and
        backward channels are estimated following the steps in
        Section 4 [1]_. See `variance_stokes_constant()`.
        A Normal distribution is assigned to each
        intensity measurement that is centered at the measurement and using the
        estimated variance. Second, a multi-variate Normal distribution is
        assigned to the estimated parameters using the covariance matrix from
        the calibration procedure presented in Section 6 [1]_ (`p_cov`). Third,
        Normal distributions are assigned for :math:`A` (`ds.alpha`)
        for each location
        outside of the reference sections. These distributions are centered
        around :math:`A_p` and have variance :math:`\sigma^2\left[A_p\right]`
        given by Equations 44 and 45. Fourth, the distributions are sampled
        and :math:`T_{\mathrm{F},m,n}` and :math:`T_{\mathrm{B},m,n}` are
        computed with Equations 16 and 17, respectively. Fifth, step four is repeated to
        compute, e.g., 10,000 realizations (`mc_sample_size`) of :math:`T_{\mathrm{F},m,n}` and
        :math:`T_{\mathrm{B},m,n}` to approximate their probability density
        functions. Sixth, the standard uncertainties of
        :math:`T_{\mathrm{F},m,n}` and :math:`T_{\mathrm{B},m,n}`
        (:math:`\sigma\left[T_{\mathrm{F},m,n}\right]` and
        :math:`\sigma\left[T_{\mathrm{B},m,n}\right]`) are estimated with the
        standard deviation of their realizations. Seventh, for each realization
        :math:`i` the temperature :math:`T_{m,n,i}` is computed as the weighted
        average of :math:`T_{\mathrm{F},m,n,i}` and
        :math:`T_{\mathrm{B},m,n,i}`:

        .. math::

            T_{m,n,i} =\
            \sigma^2\left[T_{m,n}\right]\left({\frac{T_{\mathrm{F},m,n,i}}{\
            \sigma^2\left[T_{\mathrm{F},m,n}\right]} +\
            \frac{T_{\mathrm{B},m,n,i}}{\
            \sigma^2\left[T_{\mathrm{B},m,n}\right]}}\right)

        where

        .. math::

            \sigma^2\left[T_{m,n}\right] = \frac{1}{1 /\
            \sigma^2\left[T_{\mathrm{F},m,n}\right] + 1 /\
            \sigma^2\left[T_{\mathrm{B},m,n}\right]}

        The best estimate of the temperature :math:`T_{m,n}` is computed
        directly from the best estimates of :math:`T_{\mathrm{F},m,n}` and
        :math:`T_{\mathrm{B},m,n}` as:

        .. math::
            T_{m,n} =\
            \sigma^2\left[T_{m,n}\right]\left({\frac{T_{\mathrm{F},m,n}}{\
            \sigma^2\left[T_{\mathrm{F},m,n}\right]} + \frac{T_{\mathrm{B},m,n}}{\
            \sigma^2\left[T_{\mathrm{B},m,n}\right]}}\right)

        Alternatively, the best estimate of :math:`T_{m,n}` can be approximated
        with the mean of the :math:`T_{m,n,i}` values. Finally, the 95\%
        confidence interval for :math:`T_{m,n}` are estimated with the 2.5\% and
        97.5\% percentiles of :math:`T_{m,n,i}`.

        Assumes sections are set.

        Parameters
        ----------
        p_val : array-like, optional
            Define `p_val`, `p_var`, `p_cov` if you used an external function
            for calibration. Has size `1 + 2 * nt + nx + 2 * nt * nta`.
            First value is :math:`\gamma`, then `nt` times
            :math:`D_\mathrm{F}`, then `nt` times
            :math:`D_\mathrm{B}`, then for each location :math:`D_\mathrm{B}`,
            then for each connector that introduces directional attenuation two
            parameters per time step.
        p_cov : array-like, optional
            The covariances of `p_val`. Square matrix.
            If set to False, no uncertainty in the parameters is propagated
            into the confidence intervals. Similar to the spec sheets of the DTS
            manufacturers. And similar to passing an array filled with zeros.
        st_var, ast_var, rst_var, rast_var : float, callable, array-like, optional
            The variance of the measurement noise of the Stokes signals in the
            forward direction. If `float` the variance of the noise from the
            Stokes detector is described with a single value.
            If `callable` the variance of the noise from the Stokes detector is
            a function of the intensity, as defined in the callable function.
            Or manually define a variance with a DataArray of the shape
            `ds.st.shape`, where the variance can be a function of time and/or
            x. Required if method is wls.
        conf_ints : iterable object of float
            A list with the confidence boundaries that are calculated. Valid
            values are between [0, 1].
        mc_sample_size : int
            Size of the monte carlo parameter set used to calculate the
            confidence interval
        var_only_sections : bool
            useful if using the ci_avg_x_flag. Only calculates the var over the
            sections, so that the values can be compared with accuracy along the
            reference sections. Where the accuracy is the variance of the
            residuals between the estimated temperature and temperature of the
            water baths.
        da_random_state
            For testing purposes. Similar to random seed. The seed for dask.
            Makes random not so random. To produce reproducable results for
            testing environments.
        mc_remove_set_flag : bool
            Remove the monte carlo data set, from which the CI and the
            variance are calculated.
        reduce_memory_usage : bool
            Use less memory but at the expense of longer computation time

        Returns:
        --------
        dict

        References:
        -----------
        .. [1] des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation
            of Temperature and Associated Uncertainty from Fiber-Optic Raman-
            Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235.
            https://doi.org/10.3390/s20082235

        TODO: Use get_params_from_pval_double_ended() to extract parameter sets from mc
        """
        pass

    def average_monte_carlo_single_ended(
        self,
        result,
        st_var,
        ast_var,
        conf_ints=None,
        mc_sample_size=100,
        ci_avg_time_flag1=False,
        ci_avg_time_flag2=False,
        ci_avg_time_sel=None,
        ci_avg_time_isel=None,
        ci_avg_x_flag1=False,
        ci_avg_x_flag2=False,
        ci_avg_x_sel=None,
        ci_avg_x_isel=None,
        da_random_state=None,
        mc_remove_set_flag=True,
        reduce_memory_usage=False,
    ):
        """Average temperatures from single-ended setups.

        Four types of averaging are implemented. Please see Example Notebook 16.

        Parameters
        ----------
        result : xr.Dataset
            The result from the `calibrate_single_ended()` method.
        st_var, ast_var : float, callable, array-like
            The variance of the measurement noise of the Stokes signals in the
            forward direction. If `float` the variance of the noise from the
            Stokes detector is described with a single value.
            If `callable` the variance of the noise from the Stokes detector is
            a function of the intensity, as defined in the callable function.
            Or manually define a variance with a DataArray of the shape
            `ds.st.shape`, where the variance can be a function of time and/or
            x. Required if method is wls.
        conf_ints : iterable object of float
            A list with the confidence boundaries that are calculated. Valid
            values are between
            [0, 1].
        mc_sample_size : int
            Size of the monte carlo parameter set used to calculate the
            confidence interval
        ci_avg_time_flag1 : bool
            The confidence intervals differ each time step. Assumes the
            temperature varies during the measurement period. Computes the
            arithmic temporal mean. If you would like to know the confidence
            interfal of:
            (1) a single additional measurement. So you can state "if another
            measurement were to be taken, it would have this ci"
            (2) all measurements. So you can state "The temperature remained
            during the entire measurement period between these ci bounds".
            Adds "tmpw" + '_avg1' and "tmpw" + '_mc_avg1_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avg1` are added to the DataStore. Works independently of the
            ci_avg_time_flag2 and ci_avg_x_flag.
        ci_avg_time_flag2 : bool
            The confidence intervals differ each time step. Assumes the
            temperature remains constant during the measurement period.
            Computes the inverse-variance-weighted-temporal-mean temperature
            and its uncertainty.
            If you would like to know the confidence interfal of:
            (1) I want to estimate a background temperature with confidence
            intervals. I hereby assume the temperature does not change over
            time and average all measurements to get a better estimate of the
            background temperature.
            Adds "tmpw" + '_avg2' and "tmpw" + '_mc_avg2_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avg2` are added to the DataStore. Works independently of the
            ci_avg_time_flag1 and ci_avg_x_flag.
        ci_avg_time_sel : slice
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        ci_avg_time_isel : iterable of int
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        ci_avg_x_flag1 : bool
            The confidence intervals differ at each location. Assumes the
            temperature varies over `x` and over time. Computes the
            arithmic spatial mean. If you would like to know the confidence
            interfal of:
            (1) a single additional measurement location. So you can state "if
            another measurement location were to be taken,
            it would have this ci"
            (2) all measurement locations. So you can state "The temperature
            along the fiber remained between these ci bounds".
            Adds "tmpw" + '_avgx1' and "tmpw" + '_mc_avgx1_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avgx1` are added to the DataStore. Works independently of the
            ci_avg_time_flag1, ci_avg_time_flag2 and ci_avg_x2_flag.
        ci_avg_x_flag2 : bool
            The confidence intervals differ at each location. Assumes the
            temperature is the same at each location but varies over time.
            Computes the inverse-variance-weighted-spatial-mean temperature
            and its uncertainty.
            If you would like to know the confidence interfal of:
            (1) I have put a lot of fiber in water, and I know that the
            temperature variation in the water is much smaller than along
            other parts of the fiber. And I would like to average the
            measurements from multiple locations to improve the estimated
            temperature.
            Adds "tmpw" + '_avg2' and "tmpw" + '_mc_avg2_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avg2` are added to the DataStore. Works independently of the
            ci_avg_time_flag1 and ci_avg_x_flag.
        ci_avg_x_sel : slice
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        ci_avg_x_isel : iterable of int
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        da_random_state
            For testing purposes. Similar to random seed. The seed for dask.
            Makes random not so random. To produce reproducable results for
            testing environments.
        mc_remove_set_flag : bool
            Remove the monte carlo data set, from which the CI and the
            variance are calculated.
        reduce_memory_usage : bool
            Use less memory but at the expense of longer computation time

        Returns:
        --------
        dict

        """
        pass

    def average_monte_carlo_double_ended(
        self,
        result,
        st_var,
        ast_var,
        rst_var,
        rast_var,
        conf_ints=None,
        mc_sample_size=100,
        ci_avg_time_flag1=False,
        ci_avg_time_flag2=False,
        ci_avg_time_sel=None,
        ci_avg_time_isel=None,
        ci_avg_x_flag1=False,
        ci_avg_x_flag2=False,
        ci_avg_x_sel=None,
        ci_avg_x_isel=None,
        da_random_state=None,
        mc_remove_set_flag=True,
        reduce_memory_usage=False,
        **kwargs,
    ):
        """Average temperatures from double-ended setups.

        Four types of averaging are implemented. Please see Example Notebook 16.

        Parameters
        ----------
        result : xr.Dataset
            The result from the `calibrate_double_ended()` method.
        st_var, ast_var, rst_var, rast_var : float, callable, array-like
            The variance of the measurement noise of the Stokes signals in the
            forward direction. If `float` the variance of the noise from the
            Stokes detector is described with a single value.
            If `callable` the variance of the noise from the Stokes detector is
            a function of the intensity, as defined in the callable function.
            Or manually define a variance with a DataArray of the shape
            `ds.st.shape`, where the variance can be a function of time and/or
            x. Required if method is wls.
        conf_ints : iterable object of float
            A list with the confidence boundaries that are calculated. Valid
            values are between
            [0, 1].
        mc_sample_size : int
            Size of the monte carlo parameter set used to calculate the
            confidence interval
        ci_avg_time_flag1 : bool
            The confidence intervals differ each time step. Assumes the
            temperature varies during the measurement period. Computes the
            arithmic temporal mean. If you would like to know the confidence
            interfal of:
            (1) a single additional measurement. So you can state "if another
            measurement were to be taken, it would have this ci"
            (2) all measurements. So you can state "The temperature remained
            during the entire measurement period between these ci bounds".
            Adds "tmpw" + '_avg1' and "tmpw" + '_mc_avg1_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avg1` are added to the DataStore. Works independently of the
            ci_avg_time_flag2 and ci_avg_x_flag.
        ci_avg_time_flag2 : bool
            The confidence intervals differ each time step. Assumes the
            temperature remains constant during the measurement period.
            Computes the inverse-variance-weighted-temporal-mean temperature
            and its uncertainty.
            If you would like to know the confidence interfal of:
            (1) I want to estimate a background temperature with confidence
            intervals. I hereby assume the temperature does not change over
            time and average all measurements to get a better estimate of the
            background temperature.
            Adds "tmpw" + '_avg2' and "tmpw" + '_mc_avg2_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avg2` are added to the DataStore. Works independently of the
            ci_avg_time_flag1 and ci_avg_x_flag.
        ci_avg_time_sel : slice
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        ci_avg_time_isel : iterable of int
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        ci_avg_x_flag1 : bool
            The confidence intervals differ at each location. Assumes the
            temperature varies over `x` and over time. Computes the
            arithmic spatial mean. If you would like to know the confidence
            interfal of:
            (1) a single additional measurement location. So you can state "if
            another measurement location were to be taken,
            it would have this ci"
            (2) all measurement locations. So you can state "The temperature
            along the fiber remained between these ci bounds".
            Adds "tmpw" + '_avgx1' and "tmpw" + '_mc_avgx1_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avgx1` are added to the DataStore. Works independently of the
            ci_avg_time_flag1, ci_avg_time_flag2 and ci_avg_x2_flag.
        ci_avg_x_flag2 : bool
            The confidence intervals differ at each location. Assumes the
            temperature is the same at each location but varies over time.
            Computes the inverse-variance-weighted-spatial-mean temperature
            and its uncertainty.
            If you would like to know the confidence interfal of:
            (1) I have put a lot of fiber in water, and I know that the
            temperature variation in the water is much smaller than along
            other parts of the fiber. And I would like to average the
            measurements from multiple locations to improve the estimated
            temperature.
            Adds "tmpw" + '_avg2' and "tmpw" + '_mc_avg2_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avg2` are added to the DataStore. Works independently of the
            ci_avg_time_flag1 and ci_avg_x_flag.
        ci_avg_x_sel : slice
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        ci_avg_x_isel : iterable of int
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        da_random_state
            For testing purposes. Similar to random seed. The seed for dask.
            Makes random not so random. To produce reproducable results for
            testing environments.
        mc_remove_set_flag : bool
            Remove the monte carlo data set, from which the CI and the
            variance are calculated.
        reduce_memory_usage : bool
            Use less memory but at the expense of longer computation time

        Returns:
        --------
        dict

        """
        pass
