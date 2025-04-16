import numpy as np

def kalman_filter(mu_prev, Sigma_prev, u_t, z_t, A, B, C, R, Q):
    """
    One iteration of a linear Kalman Filter for a 3D pose: (x, y, theta).
    
    Parameters
    ----------
    mu_prev : np.ndarray of shape (3, 1)
        Previous state mean [x, y, theta]^T
    Sigma_prev : np.ndarray of shape (3, 3)
        Previous state covariance
    u_t : np.ndarray of shape (3, 1)
        Control input at time t [v_x, v_y, v_theta]^T
    z_t : np.ndarray of shape (3, 1)
        Measurement at time t [x_meas, y_meas, theta_meas]^T
    A : np.ndarray of shape (3, 3)
        State transition matrix
    B : np.ndarray of shape (3, 3)
        Control matrix
    C : np.ndarray of shape (3, 3)
        Measurement matrix
    R : np.ndarray of shape (3, 3)
        Process noise covariance
    Q : np.ndarray of shape (3, 3)
        Measurement noise covariance

    Returns
    -------
    mu : np.ndarray of shape (3, 1)
        Updated state mean (posterior)
    Sigma : np.ndarray of shape (3, 3)
        Updated state covariance (posterior)
    """
    # --- Prediction ---
    mu_pred = A @ mu_prev + B @ u_t
    Sigma_pred = A @ Sigma_prev @ A.T + R

    # --- Correction ---
    # Kalman Gain
    S = C @ Sigma_pred @ C.T + Q  # Innovation covariance
    K = Sigma_pred @ C.T @ np.linalg.inv(S)

    # Innovation or residual
    y_tilde = z_t - (C @ mu_pred)

    # Updated state mean
    mu = mu_pred + K @ y_tilde

    # Updated state covariance
    I = np.eye(3)
    Sigma = (I - K @ C) @ Sigma_pred
    
    return mu, Sigma