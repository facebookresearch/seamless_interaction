from abc import abstractmethod

from torch.utils.data import Dataset


class TorchSeamlessCommunicationDataset(Dataset):
    @abstractmethod
    def __init__(self, seed=0) -> None:
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


class TorchSeamlessCommunicationMondadicDataset(Dataset):
    pass


class TorchSeamlessCommunicationDyadicDataset(Dataset):
    pass
