import pickle
import os

def save(filename, object):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(object, f)
        print(f"Trajectory saved to {filename}")
        
    except Exception as e:
        print(f"Error saving trajectory: {e}")

def load(filename):
    """Load a trajectory from a file"""
    if not os.path.exists(filename):
        print(f"File {filename} not found")
        
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
        print(f"Loaded trajectory with  waypoints from {filename}")
        
    except Exception as e:
        print(f"Error loading trajectory: {e}")