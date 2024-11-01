import pickle

# Path to the pickle file
pickle_file_path = '/home/locolinux2/.local/workspace/checkpoint/experiment_109/best_model.pkl'

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
