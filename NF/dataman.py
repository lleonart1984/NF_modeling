import torch
import numpy as np


class DataManager:
    """
    Gets the data from a dataset in a file or memory and keeps the columns representing the conditions and the targets for a specific model training.
    It can be specified a transformation mapping for a specific value in order to train.
    """
    def __init__(self, mappings, blocks):
        """
        Constructor.
        mappings: dictionary with the name of the tensor for the column and the mapping function.
        Maps function must be None if no map is needed.
        blocks: tuple with partition counting e.g. (3,4) get batches of tuples with 3 and 4 columns
        """
        self.device = torch.device('cpu')
        self.mappings = mappings  # save mappings for filtering
        self.blocks = blocks  # used to split the data columns in several blocks for conditioning, target, etc.
        self.data = torch.tensor(np.array([]))
        self.data_count = 0
        self.set_device(None)

    def load_data(self, data, limit=None):
        def get_data_and_map(data_id, map_function): return map_function(
            data[data_id].astype(np.float32)) if map_function is not None else data[data_id].astype(np.float32)
        self.data = torch.tensor([get_data_and_map(d, map_function) for d, map_function in self.mappings.items()]).T
        self.data_count = len(self.data) if limit is None else limit

    def load_file(self, path, limit=None):
        self.load_data(np.load(path), limit)

    def get_batches(self, batch_size):
        """
        Gets the data randomly placed in batches of batch_size length.
        All batches are granted to have the same size.
        If batch_size is greater than dataset size, then a single batch with all the data shuffled is return.
        """
        batch_size = min(batch_size, self.data_count)
        total_size = batch_size * int(self.data_count / batch_size)
        indices = torch.randperm(total_size)
        sums = [sum(self.blocks[: i]) for i in range(len(self.blocks) + 1)]

        return zip(*[torch.split(self.data[0:total_size, sums[i]:sums[i + 1]][indices], batch_size) for i in
                     range(0, len(self.blocks))])

    def set_device(self, device):
        """
        Moves the data to a specific device.
        If device is None, then the data is placed on CPU memory.
        """
        self.data = self.data.to(device)
        self.device = device if device is not None else torch.device('cpu')

    def get_filtered_data(self, filters):
        """
        Returns a frame of the data applying the filters for each column.
        filters is a dictionary of columns vs tuple of min max interval.
        """
        sel = torch.Tensor(len(self.data)).to(dtype=torch.bool).fill_(True).to(self.device)

        for column_index, k in enumerate(self.mappings.keys()):
            if k not in filters:  # no need to filter by that key
                continue
            a, b = filters[k]
            if self.mappings[k] is not None:
                a, b = self.mappings[k](a), self.mappings[k](b)
                a, b = min(a, b), max(a, b)

            sel = sel & (self.data[:, column_index] >= a) & (self.data[:, column_index] <= b)

        return self.data[sel]

