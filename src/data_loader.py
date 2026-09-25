class DataLoader:
    def __init__(self, train_data_path, validation_data_path):
        self.train_data_path = train_data_path
        self.validation_data_path = validation_data_path

    def load_train_data(self):
        # Load training data from the specified path
        import pandas as pd
        train_data = pd.read_csv(self.train_data_path)
        return train_data

    def load_validation_data(self):
        # Load validation data from the specified path
        import pandas as pd
        validation_data = pd.read_csv(self.validation_data_path)
        return validation_data