class FeTSDataLoader():
    """
    A data loader class for the FeTS challenge that handles training and validation data loaders.

    Attributes:
        train_loader (DataLoader): The data loader for the training dataset.
        valid_loader (DataLoader): The data loader for the validation dataset.
    """

    def __init__(self, train_loader, valid_loader):
        """
        Initializes the FeTSDataLoader with training and validation data loaders.

        Args:
            train_loader (DataLoader): The data loader for the training dataset.
            valid_loader (DataLoader): The data loader for the validation dataset.
        """
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def get_train_loader(self):
        """
        Returns the data loader for the training dataset.

        Returns:
            DataLoader: The data loader for the training dataset.
        """
        return self.train_loader

    def get_valid_loader(self):
        """
        Returns the data loader for the validation dataset.

        Returns:
            DataLoader: The data loader for the validation dataset.
        """
        return self.valid_loader

    def get_train_data_size(self):
        """
        Returns the size of the training dataset.

        Returns:
            int: The number of samples in the training dataset.
        """
        return len(self.train_loader.dataset)
    
    def get_valid_data_size(self):
        """
        Returns the size of the validation dataset.

        Returns:
            int: The number of samples in the validation dataset.
        """
        return len(self.valid_loader.dataset)