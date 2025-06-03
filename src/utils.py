import h5py
import numpy as np
from scipy import signal
import torch
import torch.nn as nn
from torch.utils.data import Dataset


##################################################
# Dataset preparation part
##################################################


class TripletDataset(Dataset):
    def __init__(self, data, labels, device_range):
        self.data = data
        self.labels = labels
        self.dev_range = device_range
        self.label_to_indices = {label: np.where(labels == label)[0] for label in device_range}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get anchor and positive
        positive_label = self.labels[index]
        anchor_idx = index

        candidate_indices = self.label_to_indices[positive_label.item()]
        candidate_indices = candidate_indices[candidate_indices != anchor_idx]
        positive_idx = np.random.choice(candidate_indices)

        # Get negative
        negative_label = np.random.choice([x for x in self.dev_range if x != positive_label])
        negative_idx = np.random.choice(self.label_to_indices[negative_label.item()])

        anchor = self.data[anchor_idx]
        positive = self.data[positive_idx]
        negative = self.data[negative_idx]

        return anchor.astype("float32"), positive.astype("float32"), negative.astype("float32")


def awgn(data, snr_range):
    """
    Add AWGN to the data.

    Args:
        data: The data to add AWGN to.
        snr_range: The SNR range to use.

    Returns:
        - data: The data with AWGN added.

    """
    num_pkts = data.shape[0]
    SNRdB = np.random.uniform(snr_range[0], snr_range[-1], num_pkts)
    for pkt_idx in range(num_pkts):
        signal = data[pkt_idx]
        # SNRdB = uniform(snr_range[0],snr_range[-1])
        SNR_linear = 10 ** (SNRdB[pkt_idx] / 10)
        P = np.sum(abs(signal) ** 2) / len(signal)
        N0 = P / SNR_linear
        noise = np.sqrt(N0 / 2) * (np.random.standard_normal(len(signal)) + 1j * np.random.standard_normal(len(signal)))
        data[pkt_idx] = signal + noise

    return data


class HDF5Loader:
    def __init__(self):
        self.dataset_name = "data"
        self.labelset_name = "label"

    def _convert_to_complex(self, data):
        """
        Convert the loaded data to complex IQ samples.

        Args:
            data: The loaded data.

        Returns:
            - data_complex: The converted complex IQ samples.

        """
        ##################################################
        # Get the number of packets and the number of IQ samples (I0, I1, ..., In, Q0, Q1, ..., Qn)
        ##################################################
        num_pkts = data.shape[0]
        num_iq_samples = data.shape[1]

        ##################################################
        # Convert the data to complex IQ samples
        ##################################################
        num_complex_samples = round(num_iq_samples / 2)
        data_complex = np.zeros([num_pkts, num_complex_samples], dtype=complex)
        data_complex = data[:, :num_complex_samples] + 1j * data[:, num_complex_samples:]
        return data_complex

    def load_complex_IQ_samples(self, file_path, dev_range, pkt_range):
        """
        Load IQ samples from a dataset.

        Args:
            file_path: The h5 dataset path.
            dev_range: The loaded device range.
            pkt_range: The loaded packets range.

        Returns:
            A tuple containing:
            - data: The loaded complex IQ samples.
            - labels: The true label of each received packet.

        """
        ##################################################
        # Read the h5 file
        ##################################################
        f = h5py.File(file_path, "r")

        ##################################################
        # Process label
        ##################################################
        labels = f[self.labelset_name][:]
        labels = labels.astype(int)
        labels = np.transpose(labels)
        labels = labels - 1  # Corresponding to the index of the device

        label_start = int(labels[0]) + 1
        label_end = int(labels[-1]) + 1
        num_devs = label_end - label_start + 1
        num_pkts = len(labels)
        num_pkts_per_dev = int(num_pkts / num_devs)

        print("Dataset information: Dev " + str(label_start) + " to Dev " + str(label_end) + ", " + str(num_pkts_per_dev) + " packets per device.")

        ##################################################
        # Process data, reorder the data and label
        ##################################################
        sample_indexes = []

        for dev_idx in dev_range:
            sample_indexes_dev = np.where(labels == dev_idx)[0][pkt_range].tolist()  # Get the packet indexes for the current device
            sample_indexes.extend(sample_indexes_dev)

        data = f[self.dataset_name][sample_indexes]  # Data: dev0 data, dev1 data, ..., devN data
        data = self._convert_to_complex(data)  # Convert the data to complex IQ samples

        labels = labels[sample_indexes]  # Reorder the labels

        f.close()

        return data, labels


class ChanIndSpecTransformer:
    def __init__(self):
        pass

    def _normalize(self, data):
        """
        Normalize the signal.

        Args:
            data: The IQ samples.

        Returns:
            - sig_norm: The normalized IQ samples.

        """
        sig_norm = np.zeros(data.shape, dtype=complex)

        for i in range(data.shape[0]):
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude**2))
            sig_norm[i] = data[i] / rms

        return sig_norm

    def _crop_spec(self, spectrogram):
        """
        Crop the generated channel independent spectrogram.

        Args:
            spectrogram: The generated channel independent spectrogram.

        Returns:
            - spectrogram_cropped: The cropped channel independent spectrogram.

        """
        num_row = spectrogram.shape[0]
        left = round(num_row * 0.3)
        right = round(num_row * 0.7)
        spectrogram_cropped = spectrogram[left:right]

        return spectrogram_cropped

    def _gen_chan_ind_spec(self, sig, win_len=256, overlap=128):
        """
        Converts the IQ samples to a channel independent spectrogram according to
        set window and overlap length.

        Args:
            sig: The IQ samples.
            win_len: The window length used in STFT.
            overlap: The overlap length used in STFT.

        Returns:
            - chan_ind_spec_amp: The generated channel independent spectrogram.

        """
        # Short-time Fourier transform (STFT).
        f, t, spec = signal.stft(sig, window="boxcar", nperseg=win_len, noverlap=overlap, nfft=win_len, return_onesided=False, padded=False, boundary=None)

        # FFT shift to adjust the central frequency.
        spec = np.fft.fftshift(spec, axes=0)

        # Generate channel independent spectrogram.
        chan_ind_spec = spec[:, 1:] / spec[:, :-1]

        # Take the logarithm of the magnitude.
        chan_ind_spec_amp = np.log10(np.abs(chan_ind_spec) ** 2)

        return chan_ind_spec_amp

    def convert(self, data):
        """
        Converts the IQ samples to channel independent spectrograms.

        Args:
            data: The IQ samples.

        Returns:
            - data_channel_ind_spec: The channel independent spectrograms.

        """

        # Normalize the IQ samples.
        data = self._normalize(data)

        # Calculate the size of channel independent spectrograms.
        num_samples = data.shape[0]
        num_rows = int(256 * 0.4)
        num_cols = int(np.floor((data.shape[1] - 256) / 128 + 1) - 1)
        data_channel_ind_spec = np.zeros([num_samples, num_rows, num_cols, 1])

        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in range(num_samples):
            chan_ind_spec_amp = self._gen_chan_ind_spec(data[i])
            chan_ind_spec_amp = self._crop_spec(chan_ind_spec_amp)
            data_channel_ind_spec[i, :, :, 0] = chan_ind_spec_amp

        return data_channel_ind_spec


class SpecTransformer:
    def __init__(self):
        pass

    def _normalize(self, data):
        """
        Normalize the signal.

        Args:
            data: The IQ samples.

        Returns:
            - sig_norm: The normalized IQ samples.

        """
        sig_norm = np.zeros(data.shape, dtype=complex)

        for i in range(data.shape[0]):
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude**2))
            sig_norm[i] = data[i] / rms

        return sig_norm

    def _crop_spec(self, spectrogram):
        """
        Crop the generated channel independent spectrogram.

        Args:
            spectrogram: The generated channel independent spectrogram.

        Returns:
            - spectrogram_cropped: The cropped channel independent spectrogram.

        """
        num_row = spectrogram.shape[0]
        left = round(num_row * 0.3)
        right = round(num_row * 0.7)
        spectrogram_cropped = spectrogram[left:right]

        return spectrogram_cropped

    def _gen_spec(self, sig, win_len=256, overlap=128):
        """
        Converts the IQ samples to a channel independent spectrogram according to
        set window and overlap length.

        Args:
            sig: The IQ samples.
            win_len: The window length used in STFT.
            overlap: The overlap length used in STFT.

        Returns:
            - chan_ind_spec_amp: The generated channel independent spectrogram.

        """
        # Short-time Fourier transform (STFT).
        f, t, spec = signal.stft(sig, window="boxcar", nperseg=win_len, noverlap=overlap, nfft=win_len, return_onesided=False, padded=False, boundary=None)

        # FFT shift to adjust the central frequency.
        spec = np.fft.fftshift(spec, axes=0)

        # Take the logarithm of the magnitude.
        chan_ind_spec_amp = np.log10(np.abs(spec) ** 2)

        return chan_ind_spec_amp

    def convert(self, data):
        """
        Converts the IQ samples to channel independent spectrograms.

        Args:
            data: The IQ samples.

        Returns:
            - data_channel_ind_spec: The channel independent spectrograms.

        """

        # Normalize the IQ samples.
        data = self._normalize(data)

        # Calculate the size of channel independent spectrograms.
        num_samples = data.shape[0]
        num_rows = int(256 * 0.4)
        num_cols = int(np.floor((data.shape[1] - 256) / 128 + 1))
        spec = np.zeros([num_samples, num_rows, num_cols, 1])

        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in range(num_samples):
            spec_amp = self._gen_spec(data[i])
            spec_amp = self._crop_spec(spec_amp)
            spec[i, :, :, 0] = spec_amp

        return spec


##################################################
# Training helper part
##################################################


class TripletLoss(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive).pow(2), dim=1)
        neg_dist = torch.sum((anchor - negative).pow(2), dim=1)

        loss = pos_dist - neg_dist + self.alpha
        loss = torch.clamp(loss, min=0.0)
        loss = torch.mean(loss)
        return loss


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


class LRScheduler:
    def __init__(self, optimizer, patience=10, min_lr=1e-6, factor=0.1):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=self.patience, factor=self.factor, min_lr=self.min_lr, verbose=True)

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
