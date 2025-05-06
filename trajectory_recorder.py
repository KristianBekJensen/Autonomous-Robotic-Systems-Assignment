import numpy as np
import os
import pickle

class TrajectoryRecorder:
    """
    Class for recording, saving, loading, and replaying robot trajectories
    """
    def __init__(self):
        # List to store trajectory points (time, x, y, theta, v_left, v_right)
        self.trajectory = []
        self.recording = False
        self.replaying = False
        self.replay_index = 0
        self.start_time = 0
        self.replay_start_time = 0
        
    def start_recording(self, current_time):
        """Start recording a new trajectory"""
        self.trajectory = []
        self.recording = True
        self.start_time = current_time
        print("Recording started")
        
    def stop_recording(self):
        """Stop recording the trajectory"""
        self.recording = False
        print(f"Recording stopped. Recorded {len(self.trajectory)} waypoints.")
        
    def record_point(self, time, x, y, theta, v_left, v_right):
        """Record a single trajectory point"""
        if self.recording:
            # Store time as relative to start_time
            relative_time = time - self.start_time
            self.trajectory.append((relative_time, x, y, theta, v_left, v_right))
    
    def save_trajectory(self, filename="trajectory.pkl"):
        """Save the recorded trajectory to a file"""
        if not self.trajectory:
            print("No trajectory to save")
            return False
            
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.trajectory, f)
            print(f"Trajectory saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving trajectory: {e}")
            return False
    
    def load_trajectory(self, filename="trajectory.pkl"):
        """Load a trajectory from a file"""
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return False
            
        try:
            with open(filename, 'rb') as f:
                self.trajectory = pickle.load(f)
            print(f"Loaded trajectory with {len(self.trajectory)} waypoints from {filename}")
            return True
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            return False
    
    def start_replay(self, current_time):
        """Start replaying the loaded trajectory"""
        if not self.trajectory:
            print("No trajectory to replay")
            return False
            
        self.replaying = True
        self.replay_index = 0
        self.replay_start_time = current_time
        print("Replay started")
        return True
    
    def stop_replay(self):
        """Stop replaying the trajectory"""
        self.replaying = False
        print("Replay stopped")
    
    def get_replay_point(self, current_time):
        """
        Get the appropriate trajectory point for the current time
        Returns (x, y, theta, v_left, v_right) or None if replay is complete
        """
        if not self.replaying or not self.trajectory:
            return None
            
        replay_time = current_time - self.replay_start_time
        
        # Find the trajectory point that corresponds to the current time
        while self.replay_index < len(self.trajectory) - 1:
            # If we're before the next point, interpolate between current and next
            if self.trajectory[self.replay_index + 1][0] > replay_time:
                break
            self.replay_index += 1
        
        # If we've reached the end of the trajectory
        if self.replay_index >= len(self.trajectory) - 1:
            if replay_time > self.trajectory[-1][0]:
                self.stop_replay()
                return None
        
        # Get current point
        current_point = self.trajectory[self.replay_index]
        
        # If we're at the last point or exactly at a recorded time, return that point
        if self.replay_index == len(self.trajectory) - 1 or current_point[0] == replay_time:
            return current_point[1:]  # x, y, theta, v_left, v_right
        
        # Otherwise, interpolate between current and next point
        next_point = self.trajectory[self.replay_index + 1]
        
        # Calculate interpolation factor
        t1, t2 = current_point[0], next_point[0]
        alpha = (replay_time - t1) / (t2 - t1)
        
        # Linearly interpolate position and wheel speeds
        x = current_point[1] + alpha * (next_point[1] - current_point[1])
        y = current_point[2] + alpha * (next_point[2] - current_point[2])
        
        # Interpolate angle carefully (handle wrap-around)
        theta1, theta2 = current_point[3], next_point[3]
        # Ensure the angle difference is in the range [-pi, pi]
        angle_diff = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
        theta = (theta1 + alpha * angle_diff) % (2 * np.pi)
        
        # Linearly interpolate wheel speeds
        v_left = current_point[4] + alpha * (next_point[4] - current_point[4])
        v_right = current_point[5] + alpha * (next_point[5] - current_point[5])
        
        return x, y, theta, v_left, v_right
    
    def is_recording(self):
        """Check if recording is active"""
        return self.recording
    
    def is_replaying(self):
        """Check if replay is active"""
        return self.replaying