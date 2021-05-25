# Provided by the FeTS Initiative (www.fets.ai) as part of the FeTS Challenge 2021

# Contributing Authors (alphabetical):
# Micah Sheller (Intel)


from fets.data.pytorch.gandlf_data import GANDLFData


# simply translates between data loader interfaces
class FeTSChallengeDataLoader(GANDLFData):

    def get_train_loader(self, batch_size=None, num_batches=None):
        return super().get_train_loader()

    def get_valid_loader(self, batch_size=None):
        return self.get_val_loader()

    def get_train_data_size(self):
        return self.get_training_data_size()

    def get_valid_data_size(self):
        return self.get_validation_data_size()
