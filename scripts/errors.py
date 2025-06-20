class DatasetError(Exception):
    """Exception raised for errors in the underlying data in the dataset."""

    pass


class ConfigurationError(Exception):
    """Exception raised for errors in the dataloader/dataset configuration."""

    pass
