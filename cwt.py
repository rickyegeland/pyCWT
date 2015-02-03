import numpy as np
from scipy.fftpack import fft, ifft, fftshift
from scipy.integrate import trapz
from scipy.integrate import quad
from copy import deepcopy
import matplotlib.pyplot as plt

__all__ = ['SDG', 'Morlet']

class MotherWavelet(object):
    """Class for MotherWavelets.

    Contains methods related to mother wavelets.  Also used to ensure that new
    mother wavelet objects contain the minimum requirements to be used in the
    cwt related functions.

    """

    def init_scales(self, t, scales=None, pad_to=None):
        """Initialize the scales and other values derived from the time axis.

        This method must be implemented by all subclasses of MotherWavelet
        """

        raise NotImplementedError('init_scales() needs to be implemented for the mother wavelet')

    @staticmethod
    def get_coefs(self):
        """Raise error if method for calculating mother wavelet coefficients is
        missing!

        """

        raise NotImplementedError('get_coefs() needs to be implemented for the mother wavelet')

    @staticmethod
    def get_coi_coef(sampf):
        """Raise error if Cone of Influence coefficient is not set in
        subclass wavelet. To follow the convention in the literature, please define your
        COI coef as a function of period, not scale - this will ensure
        compatibility with the scalogram method.

        """

        raise NotImplementedError('get_coi_coef() needs to be implemented in subclass wavelet')

    #add methods for computing cone of influence and mask
    def get_coi(self):
        """Compute cone of influence."""

        y1 =  self.coi_coef * np.arange(0, self.len_signal / 2)
        y2 = -self.coi_coef * np.arange(0, self.len_signal / 2) + y1[-1]
        coi = np.r_[y1, y2]
        self.coi = coi
        return coi

    def get_mask(self):
        """Get mask for cone of influence.

        Sets self.mask as an array of bools for use in np.ma.array('', mask=mask)

        """

        mask = np.ones(self.coefs.shape)
        masks = self.coi_coef * self.scales
        for s in range(0, len(self.scales)):
            if (s != 0) and (int(np.ceil(masks[s])) < mask.shape[1]):
                mask[s,np.ceil(int(masks[s])):-np.ceil(int(masks[s]))] = 0
        self.mask = mask.astype(bool)
        return self.mask

    def cwt(self, t, y, scales=None, pad_to=None, limit_coi=True, weighting_function=lambda x: x**(-0.5), deep_copy=True):
        """Computes the continuous wavelet transform of y

        This function computes the continuous wavelet transform of y

        The cwt is defined as:

            T(a,b) = w(a) integral(-inf,inf)(y(t) * psi*{(t-b)/a} dt

        which is a convolution.  In this algorithm, the convolution in the time
        domain is implemented as a multiplication in the frequency domain.

        Parameters
        ----------
        t : 1D array
            Time axis of time series to be transformed by the cwt
    
        y : 1D array
            Data axis of time series to be transformed by the cwt

        scales : array
            Array of scales used to initialize the mother wavelet.  If
            None, a set of scales will be calculated covering time
            scales represented in the time axis.

        pad_to : int
            Pad time series to a total length `pad_to` using zero
            padding.  This is used in the fft function when performing
            the convolution of the wavelet and mother wavelet in
            Fourier space.
    
        limit_coi : bool
            If True (default), and scales is None (default), then the
            range of scales automatically calculated will only extend
            to the limits defined by the cone of influence.
            Otherwise, scales up to the duration of the time axis t
            will be covered.
    
        weighting_function:  Function used to weight
            Typically w(a) = a^(-0.5) is chosen as it ensures that the
            wavelets at every scale have the same energy.

        deep_copy : bool
            If true (default), the mother wavelet object used in the creation of
            the wavelet object will be fully copied and accessible through
            wavelet.mother; if false, wavelet.mother will be a
            reference to the mother object (that is, if you change the
            mother wavelet object, you will see the changes when accessing the
            mother wavelet through the wavelet object - this is NOT good for
            tracking how the wavelet transform was computed, but setting
            deep_copy to False will save memory).

        Returns
        -------
        Returns an instance of the Wavelet class.  The coefficients of the transform
        can be obtain by the coefs() method (i.e.  wavelet.coefs() )

        Examples
        --------
        Create instance of the Morlet mother wavelet, perform the
        continuous wavelet transform and plot the scalogram.

        # import numpy as np
        # import cwt
        #
        # P = 17.0 # period
        # D = 150.0 # duration of time series, same units as P
        # N = 10 * D # number of samples in time series
        # t = np.linspace(0, D, N)
        # y = np.sin(2 * np.pi * t / P)
        #
        # wavelet = Morlet()
        # result = wavelet.cwt(t, y)
        # result.scalogram()

        References
        ----------
        Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
          and Francis Group, New York/London. 353 pp.

        """
    
        signal_dtype = y.dtype
    
        # Initialize the scales
        self.init_scales(t, scales=scales, pad_to=pad_to, limit_coi=limit_coi)

        # Transform the signal and mother wavelet into the Fourier domain
        yf=fft(y, self.len_wavelet)
        mwf=fft(self.coefs.conj(), axis=1)

        # Convolve (multiply in Fourier space)
        wt_tmp=ifft(mwf*yf[np.newaxis,:], axis=1)

        # shift output from ifft and multiply by weighting function
        wt = fftshift(wt_tmp,axes=[1]) * weighting_function(self.scales[:, np.newaxis])

        # if mother wavelet and signal are real, only keep real part of transform
        wt=wt.astype(np.lib.common_type(self.coefs, y))

        return WaveletResult(wt,self,weighting_function,signal_dtype,deep_copy)

    def ccwt(self, t, y1, y2, **kwargs):
        """Compute the continuous cross-wavelet transform of 'y1' and 'y2'

        Parameters
        ----------
        t : 1D array
            Common time axis for y1 and y2
    
        y1, y2 : 1D array, 1D array
            Data axes of two time series

        Any keyword arguments are passed to cwt().
    
        Returns
        -------
        Returns an instance of the WaveletResult class.

        """
        cwt1 = self.cwt(t, y1, **kwargs)
        cwt2 = self.cwt(t, y2, **kwargs)
        cwt2.coefs = cwt1.coefs * np.conjugate(cwt2.coefs)
        return cwt2

class SDG(MotherWavelet):
    """Class for the SDG MotherWavelet (a subclass of MotherWavelet).

    SDG(self, len_signal = None, pad_to = None, scales = None, sampf = 1,
        normalize = True, fc = 'bandpass')

    Parameters
    ----------
    len_signal : int
        Length of time series to be decomposed.

    pad_to : int
        Pad time series to a total length `pad_to` using zero padding (note,
        the signal will be zero padded automatically during continuous wavelet
        transform if pad_to is set). This is used in the fft function when
        performing the convolution of the wavelet and mother wavelet in Fourier
        space.

    scales : array
        Array of scales used to initialize the mother wavelet.

    sampf : float
        Sample frequency of the time series to be decomposed.

    normalize : bool
        If True, the normalized version of the mother wavelet will be used (i.e.
        the mother wavelet will have unit energy).

    fc : string
        Characteristic frequency - use the 'bandpass' or 'center' frequency of
        the Fourier spectrum of the mother wavelet to relate scale to period
        (default is 'bandpass').

    Returns
    -------
    Returns an instance of the MotherWavelet class which is used in the cwt and
    icwt functions.

    Examples
    --------
    Create instance of SDG mother wavelet, normalized, using 10 scales and the
    center frequency of the Fourier transform as the characteristic frequency.
    Then, perform the continuous wavelet transform and plot the scalogram.

    # x = numpy.arange(0,2*numpy.pi,numpy.pi/8.)
    # data = numpy.sin(x**2)
    # scales = numpy.arange(10)
    #
    # mother_wavelet = SDG(len_signal = len(data), scales = np.arange(10),normalize = True, fc = 'center')
    # wavelet = cwt(data, mother_wavelet)
    # wave_coefs.scalogram()

    Notes
    -----
    None

    References
    ----------
    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
      and Francis Group, New York/London. 353 pp.

    """

    def __init__(self,len_signal=None,pad_to=None,scales=None,sampf=1,normalize=True, fc = 'bandpass'):
        """Initilize SDG mother wavelet"""

        self.name='second degree of a Gaussian (mexican hat)'
        self.sampf = sampf
        self.scales = scales
        self.len_signal = len_signal
        self.normalize = normalize

        #set total length of wavelet to account for zero padding
        if pad_to is None:
            self.len_wavelet = len_signal
        else:
            self.len_wavelet = pad_to

        #set admissibility constant
        if normalize:
            self.cg = 4 * np.sqrt(np.pi) / 3.
        else:
            self.cg = np.pi

        #define characteristic frequency
        if fc is 'bandpass':
            self.fc = np.sqrt(5./2.) * self.sampf/(2 * np.pi)
        elif fc is 'center':
            self.fc = np.sqrt(2.) * self.sampf / (2 * np.pi)
        else:
            raise CharacteristicFrequencyError("fc = %s not defined"%(fc,))

        # coi_coef defined under the assumption that period is used, not scale
        self.coi_coef = 2 * np.pi * np.sqrt(2. / 5.) * self.fc # Torrence and
                                                               # Compo 1998

        # compute coefficients for the dilated mother wavelet
        self.coefs = self.get_coefs()

    def get_coefs(self):
        """Calculate the coefficients for the SDG mother wavelet"""

        # Create array containing values used to evaluate the wavelet function
        xi=np.arange(-self.len_wavelet / 2., self.len_wavelet / 2.)

        # find mother wavelet coefficients at each scale
        xsd = -xi * xi / (self.scales[:,np.newaxis] * self.scales[:,np.newaxis])

        if self.normalize is True:
            c=2. / (np.sqrt(3) * np.power(np.pi, 0.25))
        else:
            c=1.

        mw = c * (1. + xsd) * np.exp(xsd / 2.)

        self.coefs = mw

        return mw

class Morlet(MotherWavelet):
    """Class for the Morlet MotherWavelet (a subclass of MotherWavelet).

    Morlet(self, scales = None, f0 = 0.849)

    Parameters
    ----------
    f0 : float
        Central frequency of the Morlet mother wavelet.  The Fourier spectrum of
        the Morlet wavelet appears as a Gaussian centered on f0.  f0 defaults
        to a value of 0.849 (the angular frequency would be ~5.336).

    Returns
    -------
    Returns an instance of the MotherWavelet class which is used in the cwt
    and icwt functions.

    Examples
    --------
    Create instance of Morlet mother wavelet using 10 scales, perform the
    continuous wavelet transform, and plot the resulting scalogram.

    # x = numpy.arange(0,2*numpy.pi,numpy.pi/8.)
    # data = numpy.sin(x**2)
    # scales = numpy.arange(10)
    #
    # mother_wavelet = Morlet(len_signal=len(data), scales = np.arange(10))
    # wavelet = cwt(data, mother_wavelet)
    # wave_coefs.scalogram()

    Notes
    -----
    * Morlet wavelet is defined as having unit energy, so the `normalize` flag
      will always be set to True.

    * The Morlet wavelet will always use f0 as it's characteristic frequency, so
      fc is set as f0.

    References
    ----------
    Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
      and Francis Group, New York/London. 353 pp.

    """

    def __init__(self, normalize=True, f0=0.849):
        """Initilize Morlet mother wavelet."""

        self.normalize = True
        self.name = 'Morlet'

        # define characteristic frequency
        self.fc = f0

        # set admissibility constant
        # based on the simplified Morlet wavelet energy spectrum
        # in Addison (2002), eqn (2.39) - should be ok for f0 >0.84
        # FIXED using quad 04/01/2011
        #f = np.arange(0.001, 50, 0.001)
        #y = 2. * np.sqrt(np.pi) * np.exp(-np.power((2. * np.pi * f -
        #    2. * np.pi * self.fc), 2))
        #self.cg =  trapz(y[1:] / f[1:]) * (f[1]-f[0])
        self.cg = quad(lambda x : 2. * np.sqrt(np.pi) * np.exp(-np.power((2. *
                       np.pi * x - 2. * np.pi * f0), 2)), -np.Inf, np.Inf)[0]

    def init_scales(self, t, scales=None, pad_to=None, limit_coi=True):
        """Initialize the scales and other values derived from the time axis"""
        N = t.size
        T = t[-1] - t[0]
        self.len_signal = N
        self.sampf = 1. * N/T

        # set total length of wavelet to account for zero padding
        self.pad_to = pad_to
        if self.pad_to is None:
            self.len_wavelet = self.len_signal
        else:
            self.len_wavelet = self.pad_to

        # Cone of influence coefficient
        # See Torrence and Compo 1998 Fortran code:
        # http://paos.colorado.edu/research/wavelets/wave_fortran/wavelet.f
        w0 = 2. * np.pi * self.fc # angular frequency
        fourier_factor = 4. * np.pi / (w0 + np.sqrt(2. + w0**2))
        self.coi_coef = fourier_factor / (self.sampf * np.sqrt(2))

        # Initialize the scales
        if scales is None:
            if limit_coi:
                max_period = np.max(self.get_coi())
                max_period = np.min([max_period, T])
            else:
                # compute Nscales to scan from P=0 to P=T
                max_period = T
            max_scale = max_period * self.fc * self.sampf
            min_period = 1 / self.sampf
            min_scale = min_period * self.fc * self.sampf # == self.fc
            delta_scale = 0.125 # TODO: set in class
            self.scales = np.arange(min_scale, max_scale, delta_scale)
        else:
            self.scales = scales

        # compute coefficients for the dilated mother wavelet
        self.coefs = self.get_coefs()

    def get_coefs(self):
        """Calculate the coefficients for the Morlet mother wavelet."""

        # Create array containing values used to evaluate the wavelet function
        xi=np.arange(-self.len_wavelet / 2., self.len_wavelet / 2.)

        # find mother wavelet coefficients at each scale
        xsd = xi / (self.scales[:,np.newaxis])

        mw = np.power(np.pi,-0.25) * \
                     (np.exp(np.complex(1j) * 2. * np.pi * self.fc * xsd) - \
                     np.exp(-np.power((2. * np.pi * self.fc), 2) / 2.)) *  \
                     np.exp(-np.power(xsd, 2) / 2.)

        self.coefs = mw

        return mw

class WaveletResult(object):
    """Result of a continuous wavelet transform

    The WaveletResult object holds the wavelet coefficients as well as
    information on how they were obtained.

    """

    def __init__(self, wt, mother, weighting_function, signal_dtype, deep_copy=True):
        """Initialization of WaveletResult object.

        Parameters
        ----------
        coefs : array
            Array of wavelet coefficients.

        mother : object
            Mother wavelet object used in the creation of `coefs`.

        weighting_function : function
            Function used in the creation of `coefs`.

        signal_dtype : dtype
            dtype of signal used in the creation of `coefs`.

        deep_copy : bool
            If true (default), the mother wavelet object used in the creation of
            the wavelet object will be fully copied and accessible through
            wavelet.mother; if false, wavelet.mother will be a
            reference to the MotherWavelet object (that is, if you change the
            mother wavelet object, you will see the changes when accessing the
            mother wavelet through the WaveletResult object - this is NOT good for
            tracking how the wavelet transform was computed, but setting
            deep_copy to False will save memory).

        Returns
        -------
        Returns an instance of the Wavelet class.

        """

        self.coefs = wt[:,0:mother.len_signal]

        if mother.len_signal !=  mother.len_wavelet:
            self._pad_coefs = wt[:,mother.len_signal:]
        else:
            self._pad_coefs = None
        if deep_copy:
            self.mother = deepcopy(mother)
        else:
            self.mother = mother

        self.weighting_function = weighting_function
        self._signal_dtype = signal_dtype

    def icwt(self):
        """Compute the inverse continuous wavelet transform of this result.

        Examples
        --------
        Use the Morlet mother wavelet to perform wavelet transform on 'data', then
        use icwt to compute the inverse wavelet transform to come up with an estimate
        of data ('data2').  Note that data2 is not exactly equal data.

        # import matplotlib.pyplot as plt
        # from scipy.signal import SDG, Morlet, cwt, icwt, fft, ifft
        # import numpy as np
        #
        # x = np.arange(0,2*np.pi,np.pi/64)
        # data = np.sin(8*x)
        # scales=np.arange(0.5,17)
        #
        # mother_wavelet = Morlet(len_signal = len(data), scales = scales)
        # wave_coefs=cwt(data, mother_wavelet)
        # data2 = icwt(wave_coefs)
        #
        # plt.plot(data)
        # plt.plot(data2)
        # plt.show()

        References
        ----------
        Addison, P. S., 2002: The Illustrated Wavelet Transform Handbook.  Taylor
          and Francis Group, New York/London. 353 pp.

        """

        # if original wavelet was created using padding, make sure to include
        #   information that is missing after truncation (see self.coefs under __init__
        #   in class WaveletResult.
        if self.mother.len_signal !=  self.mother.len_wavelet:
            full_wc = np.c_[self.coefs, self._pad_coefs]
        else:
            full_wc = self.coefs

        # get wavelet coefficients and take fft
        wcf = fft(full_wc, axis=1)

        # get mother wavelet coefficients and take fft
        mwf = fft(self.mother.coefs, axis=1)
    
        # perform inverse continuous wavelet transform and make sure the result is the same type
        #  (real or complex) as the original data used in the transform
        x = (1. / self.mother.cg) * trapz(
            fftshift(ifft(wcf * mwf,axis=1),axes=[1]) /
            (self.mother.scales[:,np.newaxis]**2),
            dx = 1. / self.mother.sampf, axis=0)

        return x[0:self.mother.len_signal].astype(self._signal_dtype)

    def periods(self):
        """Return a period axis"""
        return self.mother.scales / self.mother.fc / self.mother.sampf

    def get_gws(self):
        """Calculate Global Wavelet Spectrum.

        References
        ----------
        Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavelet
          Analysis.  Bulletin of the American Meteorological Society, 79, 1,
          pp. 61-78.

        """

        gws = self.get_wavelet_var()

        return gws


    def get_wes(self):
        """Calculate Wavelet Energy Spectrum.

        References
        ----------
        Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavelet
          Analysis.  Bulletin of the American Meteorological Society, 79, 1,
          pp. 61-78.

        """

        coef = 1. / (self.mother.fc * self.mother.cg)

        wes = coef * trapz(np.power(np.abs(self.coefs), 2), axis = 1);

        return wes

    def get_wps(self):
        """Calculate Wavelet Power Spectrum.

        References
        ----------
        Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavelet
          Analysis.  Bulletin of the American Meteorological Society, 79, 1,
          pp. 61-78.

        """

        wps =  (1./ self.mother.len_signal) * self.get_wes()

        return wps

    def get_wavelet_var(self):
        """Calculate Wavelet Variance (a.k.a. the Global Wavelet Spectrum of
        Torrence and Compo (1998)).

        References
        ----------
        Torrence, C., and G. P. Compo, 1998: A Practical Guide to Wavelet
          Analysis.  Bulletin of the American Meteorological Society, 79, 1,
          pp. 61-78.

        """

        coef =  self.mother.cg * self.mother.fc

        wvar = (coef / self.mother.len_signal) * self.get_wes()

        return wvar

    def scalogram(self, show_coi=False, show_wps=False, ts=None, time=None,
                  use_period=True, ylog_base=None, xlog_base=None,
                  origin='top', figname=None):
        """ Scalogram plotting routine.

        Creates a simple scalogram, with optional wavelet power spectrum and
        time series plots of the transformed signal.

        Parameters
        ----------
        show_coi : bool
            Set to True to see Cone of Influence

        show_wps : bool
            Set to True to see the Wavelet Power Spectrum

        ts : array
            1D array containing time series data used in wavelet transform.  If set,
            time series will be plotted.

        time : array of datetime objects
            1D array containing time information

        use_period : bool
            Set to True to see figures use period instead of scale

        ylog_base : float
            If a log scale is desired, set `ylog_base` as float. (for log 10, set
            ylog_base = 10)

        xlog_base : float
            If a log scale is desired, set `xlog_base` as float. (for log 10, set
            xlog_base = 10) *note that this option is only valid for the wavelet power
            spectrum figure.

        origin : 'top' or 'bottom'
            Set origin of scale axis to top or bottom of figure

        Returns
        -------
        None

        Examples
        --------
        Create instance of SDG mother wavelet, normalized, using 10 scales and the
        center frequency of the Fourier transform as the characteristic frequency.
        Then, perform the continuous wavelet transform and plot the scalogram.

        # x = numpy.arange(0,2*numpy.pi,numpy.pi/8.)
        # data = numpy.sin(x**2)
        # scales = numpy.arange(10)
        #
        # mother_wavelet = SDG(len_signal = len(data), scales = np.arange(10), normalize = True, fc = 'center')
        # wavelet = cwt(data, mother_wavelet)
        # wave_coefs.scalogram(origin = 'bottom')

        """

        if ts is not None:
            show_ts = True
        else:
            show_ts = False

        if not show_wps and not show_ts:
            # only show scalogram
            figrow = 1
            figcol = 1
        elif show_wps and not show_ts:
            # show scalogram and wps
            figrow = 1
            figcol = 4
        elif not show_wps and show_ts:
            # show scalogram and ts
            figrow = 2
            figcol = 1
        else:
            # show scalogram, wps, and ts
            figrow = 2
            figcol = 4

        if time is None:
            x = np.arange(self.mother.len_signal)
        else:
            x = time

        if use_period:
            y = self.periods()
        else:
            y = self.mother.scales

        if figname is not None:
            fig = plt.figure(figsize=(16, 12), dpi=160)
        else:
            fig = plt.figure()

        ax1 = fig.add_subplot(figrow, figcol, 1)

        # if show wps, give 3/4 space to scalogram, 1/4 to wps
        if show_wps:
            # create temp axis at 3 or 4 col of row 1
            axt = fig.add_subplot(figrow, figcol, 3)
            # get location of axtmp and ax1
            axt_pos = axt.get_position()
            ax1_pos = ax1.get_position()
            axt_points = axt_pos.get_points()
            ax1_points = ax1_pos.get_points()
            # set axt_pos left bound to that of ax1
            axt_points[0][0] = ax1_points[0][0]
            ax1.set_position(axt_pos)
            fig.delaxes(axt)

        if show_coi:
            # coi_coef is defined using the assumption that you are using
            #   period, not scale, in plotting - this handles that behavior
            if not use_period:
                coi = self.mother.get_coi() / self.mother.fc / self.mother.sampf
            else:
                coi = self.mother.get_coi()

            coi[coi == 0] = y.min() - 0.1 * y.min()
            ax1.fill_between(x, np.max(y), coi, facecolor='k', alpha=0.4, zorder=2)

        contf=ax1.contourf(x,y,np.abs(self.coefs)**2)
        fig.colorbar(contf, ax=ax1, orientation='vertical', format='%2.1f')

        if ylog_base is not None:
            ax1.axes.set_yscale('log', basey=ylog_base)

        if origin is 'top':
            ax1.set_ylim((y[-1], y[0]))
        elif origin is 'bottom':
            ax1.set_ylim((y[0], y[-1]))
        else:
            raise OriginError('`origin` must be set to "top" or "bottom"')

        ax1.set_xlim((x[0], x[-1]))
        ax1.set_title('scalogram')
        ax1.set_ylabel('time')
        if use_period:
            ax1.set_ylabel('period')
            ax1.set_xlabel('time')
        else:
            ax1.set_ylabel('scales')
            if time is not None:
                ax1.set_xlabel('time')
            else:
                ax1.set_xlabel('sample')

        if show_wps:
            ax2 = fig.add_subplot(figrow,figcol,4,sharey=ax1)
            if use_period:
                ax2.plot(self.get_wps(), y, 'k')
            else:
                ax2.plot(self.mother.fc * self.get_wps(), y, 'k')

            if ylog_base is not None:
                ax2.axes.set_yscale('log', basey=ylog_base)
            if xlog_base is not None:
                ax2.axes.set_xscale('log', basey=xlog_base)
            if origin is 'top':
                ax2.set_ylim((y[-1], y[0]))
            else:
                ax2.set_ylim((y[0], y[-1]))
            if use_period:
                ax2.set_ylabel('period')
            else:
                ax2.set_ylabel('scales')
            ax2.grid()
            ax2.set_title('wavelet power spectrum')

        if show_ts:
            ax3 = fig.add_subplot(figrow, 2, 3, sharex=ax1)
            ax3.plot(x, ts)
            ax3.set_xlim((x[0], x[-1]))
            ax3.legend(['time series'])
            ax3.grid()
            # align time series fig with scalogram fig
            t = ax3.get_position()
            ax3pos=t.get_points()
            ax3pos[1][0]=ax1.get_position().get_points()[1][0]
            t.set_points(ax3pos)
            ax3.set_position(t)
            if (time is not None) or use_period:
                ax3.set_xlabel('time')
            else:
                ax3.set_xlabel('sample')

        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)
            plt.close('all')
