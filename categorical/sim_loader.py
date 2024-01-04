import pickle

import matplotlib.pyplot as plt
import numpy
from torch.utils.data import Dataset

class SimulatedShowData(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, "rb") as f:
            self.data = pickle.load(f)

        if "samples" not in self.data:
            raise RuntimeError("Pickle file did not contain a dictionary with the expected formatting.  'samples' key does not exist")

    def time(self):
        return self.data["time"]

    def visualize(self, idx):
        num_channels, _, num_time_steps = self[0].shape
        for i in idx:
            test_samples = numpy.reshape(self[i], (num_channels, num_time_steps))
            plt.figure()
            plt.plot(self.time(), test_samples[0])
            plt.plot(self.time(), test_samples[1])
            plt.plot(self.time(), test_samples[2])
        
        plt.show()

    def __getitem__(self, idx):
        return self.data["samples"][idx]

    def __len__(self):
        return self.data["samples"].shape[0]