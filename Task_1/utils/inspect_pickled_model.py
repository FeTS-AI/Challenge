import pickle
from pathlib import Path

experiment_number = 109 # change to your experiment number

pickle_file_path = Path.cwd() / '.local' / 'workspace' / 'checkpoint' / f'experiment_{experiment_number}' / 'best_model.pkl'


# Function to load and inspect the pickle file
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        print("Pickle file loaded successfully.")
        return model_data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

# Load the model
model_data = load_pickle(pickle_file_path)

# Inspect the model (print relevant information)
if model_data:
    print("Model Data: ", model_data)
