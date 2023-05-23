import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numba import jit, float64, int8
import torch
from eutils.snn_layers import threshold as thr


class BSAEncoder:

    """A class for BSA encoding algorithm."""

    def __init__(self, filter_response=None, step=1,
                 filter_amp=0.2, threshold=3, channel=64):
        """Init a BSAEncoder object.

        Parameters
        ----------
        step : int , optional.
            Default: 1.
            Steping used when moving filter on the signal.

        filter_amp : float or int, optional.
            Default: 1
            Amplitude of filter response, by increasing this parameter,
             number of spikes will decrease.

        threshold : float or int, optional.
            Default: 0.1
            Increasing the threshold, results in increased noise sensitivity.
        """

        self.channel = channel
        self.filter_amp = filter_amp
        self.step = step
        self.threshold = threshold

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, new_step):
        assert isinstance(new_step, int), "'step' should be of type 'int'."
        self._step = new_step

    @property
    def filter_amp(self):
        return self._filter_amp

    @filter_amp.setter
    def filter_amp(self, new_amp):
        assert isinstance(new_amp, (int, float)), "'filter_amp' must be of\
         types [int, float]"
        self._filter_amp = new_amp
        return

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, new_threshold):
        assert isinstance(new_threshold, (float, int)), "'threshold' should be\
         of type 'float' or 'int'."
        self._threshold = new_threshold

    def encode(self, sgnl: torch.Tensor, filter_response: torch.Tensor):
        """Encode a given signal based on init parameters.

        Parameters
        ----------
        sgnl : :obj: 'np.ndarray' , 1d array.
            Signal to be encoded.

        Notes
        -----
        The spike times will be save in self._last_spikes for later plottings.
        The encoding procedure is written in compiled mode using
         numba.jit decorator.
        """

        def calc_spike_times(
            sig: torch.Tensor,
            filter_response: torch.Tensor,
            step: int,
            threshold: float
        ):
            filter_size = filter_response.shape[1]
            sgnl_size = sig.shape[1]
            windowed_signal = [sig[:, :filter_size].clone()]
            spike_times = torch.zeros(sig.shape).to(sig.device)
            error1, error2 = [], []

            for pointer in range(0, sgnl_size, step):
                if pointer > sgnl_size - filter_size - 1:
                    break
                else:
                    error1.append(torch.sum(torch.abs(windowed_signal[-1] - filter_response), dim=1))
                    error2.append(torch.sum(torch.abs(windowed_signal[-1]), dim=1))
                    # not mult version
                    # is_spike = error1 <= error2 - threshold
                    is_spike = thr(error2[-1] - threshold - error1[-1], "gauss", 0.0)
                    windowed_signal.append(windowed_signal[-1] - filter_response * is_spike.unsqueeze(1))
                    spike_times[:, pointer] = is_spike
                    windowed_signal.append(torch.concat((windowed_signal[-1][:, step:],
                                                    sig[:, filter_size + pointer:filter_size + pointer + step]), dim=1))
            return spike_times

        spikes = calc_spike_times(sig=sgnl,
                                  filter_response=filter_response,
                                  step=self.step,
                                  threshold=float(self.threshold))
        return spikes

    def plot(
        self,
        origin: torch.Tensor,
        encoded: torch.Tensor,
        channel: int = None
    ):
        """Plot encoded version and original version of last signal."""

        fig, [ax0, ax1] = plt.subplots(nrows=2, sharex=True)
        fig.figsize = (18, 12)
        spike_times = torch.nonzero(encoded)
        if channel is not None:
            origin = origin[channel]
            spike_times = spike_times[spike_times[:, 0] == channel, 1]

        ax0.plot(origin)
        ax1.eventplot(spike_times)
        ax1.set_yticks([1])
        plt.show()

    def decode(
        self,
        origin: torch.Tensor,
        encoded: torch.Tensor,
        filter_response: torch.Tensor,
        channel: int = None
    ):
        """Decodes last encoded signal and plots two signals together."""

        spike_times = torch.nonzero(encoded)
        if channel is not None:
            origin = origin[channel]
            encoded = encoded[channel]
            spike_times = spike_times[spike_times[:, 0] == channel, 1]
            filter_response = filter_response[channel]

        decoded = torch.zeros(origin.shape)
        for spike_time in spike_times:
            decoded[spike_time: spike_time + len(filter_response)] += filter_response
        plt.plot(origin)
        # plt.plot(decoded)
        # plt.show()


        # decoded = torch.nn.functional.conv1d(encoded.unsqueeze(0).unsqueeze(0), filter_response.unsqueeze(0).unsqueeze(0), stride=self.step, padding=16)[0, 0, :origin.shape[0]]
        # plt.plot(origin)
        # plt.plot(decoded.squeeze())
        # plt.show()


# class HSAEncoder:
#     """Implementation of HSA encoding algorithm.
#
#     Init an object from this class by passing parameters, and get
#     encoded signal in term of spike times by calling 'encode' method,
#     and passing original signal to this method.
#     """
#
#     def __init__(self, filter_response=None, step=1, filter_amp=0.2):
#         """Init an encoder object.
#
#         Parameters
#         ----------
#         filter_response : :obj: 'np.ndarray' , 1d array. optional.
#             Default: A gaussian signal with M=51, std=7.
#             FIR filter as a window.
#
#         step : int , optional.
#             Default: 1.
#             Steping used when moving filter on the signal.
#
#         filter_amp : float or int, optional.
#             Default: 1
#             Amplitude of filter response, by increasing this parameter, number of spikes will decrease.
#         """
#
#         self.filter_response = filter_response
#         self.step = step
#         self.filter_amp = filter_amp
#         self._last_spike_times = None
#         self._last_signal = None
#
#     @property
#     def filter_response(self):
#         return self._filter_response
#
#     @filter_response.setter
#     def filter_response(self, new_response):
#         if new_response is None:
#             self._filter_response = signal.gaussian(M=51, std=7)
#         else:
#             assert isinstance(new_response, np.ndarray), "'filter_response' must be of np.ndarray type."
#             assert new_response.ndim == 1, "'filter_response must be a 1d array."
#             self._filter_response = new_response
#
#     @property
#     def step(self):
#         return self._step
#
#     @step.setter
#     def step(self, new_step):
#         assert isinstance(new_step, int), "'step' should be of type 'int'."
#         self._step = new_step
#
#     @property
#     def filter_amp(self):
#         return self._filter_amp
#
#     @filter_amp.setter
#     def filter_amp(self, new_amp):
#         assert isinstance(new_amp, (int, float)), "'filter_amp' must be of types [int, float]"
#         self._filter_amp = new_amp
#         self.filter_response = self.filter_response * new_amp
#         return
#
#     def encode(self, sgnl):
#         """Encode a given signal based on init parameters.
#
#         Parameters
#         ----------
#         sgnl : :obj: 'np.ndarray' , 1d array.
#             Signal to be encoded.
#
#         Notes
#         -----
#         The spike times will be save in self._last_spikes for later plottings.
#         The encoding procedure is written in compiled mode using numba.jit decorator.
#         """
#
#         @jit(int8[:](float64[:], float64[:], int8), nopython=True, cache=True)
#         def calc_spike_times(sgnl, filter_response, step):
#             filter_size = filter_response.shape[0]
#             sgnl_size = sgnl.shape[0]
#             windowed_signal = sgnl[:filter_size].copy()
#             spike_times = np.zeros(sgnl.shape, dtype=np.int8)
#             for pointer in range(0, sgnl_size, step):
#                 if pointer > sgnl_size - filter_size - 1:
#                     break
#                 else:
#                     if np.all(windowed_signal >= filter_response):
#                         windowed_signal -= filter_response
#                         spike_times[pointer] = 1
#                     windowed_signal = np.concatenate((windowed_signal[step:],
#                                                       sgnl[filter_size + pointer: filter_size + pointer + step]))
#             return spike_times
#
#         assert isinstance(sgnl, np.ndarray), "'sgnl' must be of type numpy.ndarray"
#         assert sgnl.ndim == 1, "'sgnl' must be 1d array."
#         filter_response = self.filter_response
#         step = self.step
#         self._last_signal = sgnl.copy()
#         spikes = calc_spike_times(sgnl=sgnl, filter_response=filter_response, step=step)
#         self._last_spike_times = np.where(spikes == 1)[0]
#         return spikes
#
#     def plot(self):
#         """Plot encoded version and original version of last signal."""
#
#         assert self._last_signal is not None, "You must encode at least one signal to perform plotting."
#         fig, [ax0, ax1] = plt.subplots(nrows=2, sharex=True)
#         fig.figsize = (18, 12)
#         ax0.plot(self._last_signal)
#         ax1.eventplot(self._last_spike_times)
#         ax1.set_yticks([1])
#         plt.show()
#
#     def decode(self):
#         """Decodes last encoded signal and plots two signals together."""
#
#         orig = self._last_signal
#         encoded = self._last_spike_times
#         decoded = np.zeros(orig.shape)
#         for spike_time in encoded:
#             decoded[spike_time: spike_time + len(self.filter_response)] += self.filter_response
#         plt.plot(orig)
#         plt.plot(decoded)
#         plt.show()
#
#
# class ModifiedHSA:
#     """Modified version of HSA encoding algorithm.
#
#     In this algorithm the filter response sweeps amongst whole input signal, and at every time step
#      the error is calculated. If error raised above a predefined threshold, a spike will emitted and
#      filter response is subtracted from signal at that time point."""
#
#     def __init__(self, filter_response=None, step=1, filter_amp=0.5, threshold=1):
#         """Init a ModifiedHSA encoder object.
#
#         Parameters
#         ----------
#         filter_response : :obj: 'np.ndarray' , 1d array. optional.
#             Default: A gaussian signal with M=51, std=7.
#             FIR filter as a window.
#
#         step : int , optional.
#             Default: 1.
#             Steping used when moving filter on the signal.
#
#         filter_amp : float or int, optional.
#             Default: 1
#             Amplitude of filter response, by increasing this parameter, number of spikes will decrease.
#
#         threshold : float or int, optional.
#             Default: 0.1
#             Increasing the threshold, results in increased noise sensitivity.
#         """
#
#         self.filter_response = filter_response
#         self.filter_amp = filter_amp
#         self.step = step
#         self.threshold = threshold
#         self._last_spike_times = None
#         self._last_signal = None
#
#     @property
#     def filter_response(self):
#         return self._filter_response
#
#     @filter_response.setter
#     def filter_response(self, new_response):
#         if new_response is None:
#             self._filter_response = signal.gaussian(M=51, std=7)
#         else:
#             assert isinstance(new_response, np.ndarray), "'filter_response' must be of np.ndarray type."
#             assert new_response.ndim == 1, "'filter_response must be a 1d array."
#             self._filter_response = new_response
#
#     @property
#     def step(self):
#         return self._step
#
#     @step.setter
#     def step(self, new_step):
#         assert isinstance(new_step, int), "'step' should be of type 'int'."
#         self._step = new_step
#
#     @property
#     def filter_amp(self):
#         return self._filter_amp
#
#     @filter_amp.setter
#     def filter_amp(self, new_amp):
#         assert isinstance(new_amp, (int, float)), "'filter_amp' must be of types [int, float]"
#         self._filter_amp = new_amp
#         self.filter_response = self.filter_response * new_amp
#         return
#
#     @property
#     def threshold(self):
#         return self._threshold
#
#     @threshold.setter
#     def threshold(self, new_threshold):
#         assert isinstance(new_threshold, (float, int)), "'threshold' should be of type 'float' or 'int'."
#         self._threshold = new_threshold
#
#     def encode(self, sgnl):
#         """Encode a given signal based on init parameters.
#
#         Parameters
#         ----------
#         sgnl : :obj: 'np.ndarray' , 1d array.
#             Signal to be encoded.
#
#         Notes
#         -----
#         The spike times will be save in self._last_spikes for later plottings.
#         The encoding procedure is written in compiled mode using numba.jit decorator.
#         """
#
#         @jit(int8[:](float64[:], float64[:], int8, float64), nopython=True, cache=True)
#         def calc_spike_times(sgnl, filter_response, step, threshold):
#             filter_size = filter_response.shape[0]
#             sgnl_size = sgnl.shape[0]
#             windowed_signal = sgnl[:filter_size].copy()
#             spike_times = np.zeros(sgnl.shape, dtype=np.int8)
#             for pointer in range(0, sgnl_size, step):
#                 if pointer > sgnl_size - filter_size - 1:
#                     break
#                 else:
#                     if np.any(windowed_signal <= filter_response):
#                         residuals = filter_response - windowed_signal
#                         error = residuals[residuals >= 0].sum()
#                     if error <= threshold:
#                         windowed_signal -= filter_response
#                         spike_times[pointer] = 1
#                     windowed_signal = np.concatenate((windowed_signal[step:],
#                                                       sgnl[filter_size + pointer: filter_size + pointer + step]))
#             return spike_times
#
#         assert isinstance(sgnl, np.ndarray), "'sgnl' must be of type numpy.ndarray"
#         assert sgnl.ndim == 1, "'sgnl' must be 1d array."
#         self._last_signal = sgnl.copy()
#         spikes = calc_spike_times(sgnl=sgnl, filter_response=self.filter_response,
#                                   step=self.step, threshold=float(self.threshold))
#         self._last_spike_times = np.where(spikes == 1)[0]
#         return spikes
#
#     def plot(self):
#         """Plot encoded version and original version of last signal."""
#
#         assert self._last_signal is not None, "You must encode at least one signal to perform plotting."
#         fig, [ax0, ax1] = plt.subplots(nrows=2, sharex=True)
#         fig.figsize = (18, 12)
#         ax0.plot(self._last_signal)
#         ax1.eventplot(self._last_spike_times)
#         ax1.set_xlim(-0.1 * len(self._last_signal), 1.1 * len(self._last_signal))
#         ax1.set_yticks([1])
#         plt.show()
#
#     def decode(self):
#         """Decodes last encoded signal and plots two signals together."""
#
#         orig = self._last_signal
#         encoded = self._last_spike_times
#         decoded = np.zeros(orig.shape)
#         for spike_time in encoded:
#             decoded[spike_time: spike_time + len(self.filter_response)] += self.filter_response
#         plt.plot(orig)
#         plt.plot(decoded)
#         plt.show()
