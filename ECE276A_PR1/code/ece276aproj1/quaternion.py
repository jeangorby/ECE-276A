import numpy as np
from scipy.spatial.transform import Rotation as R
import transforms3d.euler as euler
import jax.numpy as jnp
import jax
from jax import grad, jit, lax
import transforms3d.quaternions as quat


import jax.numpy as jnp


def quaternion_multiply(q, p):
    """
    Perform quaternion multiplication q * p.

    Parameters:
        q (numpy.ndarray): First quaternion (4,).
        p (numpy.ndarray): Second quaternion (4,).

    Returns:
        numpy.ndarray: Resulting quaternion (4,).
    """
    q_s, q_v = q[0], q[1:]
    p_s, p_v = p[0], p[1:]
 
    result_scalar = q_s * p_s - jnp.dot(q_v, p_v)
    result_vector = q_s * p_v + p_s * q_v + jnp.cross(q_v, p_v)

    return jnp.concatenate([jnp.array([result_scalar]), result_vector])

def quaternion_norm(q):
    return jnp.linalg.norm(q)

def quaternion_conjugate(q):
    return jnp.insert(-jnp.array(q[1:]), 0, q[0])

def quaternion_inverse(q):
    return quaternion_conjugate(q) / (quaternion_norm(q) ** 2)

def exp_map_quaternion(q):
    """
    Compute the quaternion exponential map from a quaternion input.

    Parameters:
        q (np.ndarray): Quaternion (4,) where the first element is the scalar part (q_s),
                        and the last three are the vector part (q_v).

    Returns:
        np.ndarray: Transformed quaternion (4,) using the exponential map.
    """
    assert len(q) == 4
    q_s = q[0]  # Scalar part
    q_v = q[1:]  # Vector part

    norm = jnp.linalg.norm(q_v)
    # if norm < 1e-8:
    #     return jnp.array([jnp.exp(q_s), 0, 0, 0])  # Identity quaternion for small angles
    
    exp_qs = jnp.exp(q_s)
    t1 = exp_qs * jnp.cos(norm)
    t2 = exp_qs * (q_v / norm) * (jnp.sin(norm))

    return jnp.insert(t2, 0, t1)

def log_map_quaternion(q):
    q_s = q[0]  # Scalar part
    q_v = q[1:]  # Vector part

    t1 = jnp.log(quaternion_norm(q))
    t2 = (q_v / jnp.linalg.norm(q_v)) * jnp.arccos(q_s / quaternion_norm(q))

    return jnp.insert(t2, 0, t1)


def compute_predicted_quaternions(gyro_data, timestamps): # motion model
    """
    Compute quaternion trajectory using the motion model with the exponential map.

    Parameters:
        gyro_data (numpy.ndarray): Gyroscope data (3xN) in rad/s.
        timestamps (numpy.ndarray): Corresponding timestamps (N,).

    Returns:
        numpy.ndarray: Predicted quaternion trajectory (Nx4).
    """
    n_samples = gyro_data.shape[1]
    quaternions = np.zeros((n_samples, 4))
    quaternions[0] = [1, 0, 0, 0]  # Initialize with identity quaternion

    for t in range(1, n_samples):
        dt = timestamps[t] - timestamps[t - 1]
        omega = gyro_data[:, t] 

        delta_q = exp_map_quaternion(np.insert(omega * dt / 2, 0, 0))  # Use the exponential map

        # Quaternion multiplication
        quaternions[t] = quaternion_multiply(quaternions[t - 1], delta_q)
        quaternions[t] /= np.linalg.norm(quaternions[t])  # Normalize

    return quaternions

def observation_model(q, g=jnp.array([0, 0, -9.81])):
    """
    Transform gravity vector into the IMU frame using quaternion.

    Parameters:
        q (jnp.ndarray): Quaternion (4,).
        g (jnp.ndarray): Gravity vector in the world frame (3,).

    Returns:
        jnp.ndarray: Transformed gravity vector in the IMU frame (3,).
    """
    g_quat = jnp.insert(g,0,0)  # Extend gravity vector to quaternion form
    q_inv = quaternion_inverse(q)

    # q^-1 * g * q
    rotated_g = quaternion_multiply(quaternion_multiply(q_inv, g_quat), q)
    return rotated_g  # Extract vector part


def motion_model(q, omega, dt):
    """
    Update quaternion based on angular velocity using JAX.
    """
    axis = omega * dt / 2  # Small rotation vector

    delta_q = exp_map_quaternion(jnp.insert(axis, 0,0))

    q_next = quaternion_multiply(q, delta_q)  # Use JAX-friendly quaternion multiplication

    return q_next / jnp.linalg.norm(q_next, axis=-1, keepdims=True)  # Normalize


@jit
def cost_function(q_traj, omega, accel_data, dt, g=jnp.array([0, 0, -9.81])):
    """
    Full cost function for the optimization problem.

    Parameters:
        q_traj (jnp.ndarray): Quaternion trajectory (Nx4).
        omega (jnp.ndarray): Angular velocity data (Nx3).
        accel_data (jnp.ndarray): Accelerometer data (Nx3).
        dt (jnp.ndarray): Time step data (N-1,).
        g (jnp.ndarray): Gravity vector (3,).

    Returns:
        float: Total cost.
    """
    T = len(omega)
    motion_error = 0
    observation_error = 0

    for t in range(T - 1):
        # Motion model error
        q_pred = motion_model(q_traj[t], omega[t], dt[t])
        rel_rotation = quaternion_multiply(quaternion_inverse(q_traj[t + 1]), q_pred)
        motion_error += jnp.linalg.norm(2*log_map_quaternion(rel_rotation)) ** 2

        # Observation model error
        accel_world = observation_model(q_traj[t])
        observation_error += jnp.linalg.norm(jnp.insert(accel_data[t],0,0) - accel_world) ** 2

    return 0.5 * (motion_error + observation_error)


@jit
def optimize_quaternion_trajectory(omega, accel_data, dt, num_iterations=1000, lr=0.01):
    """
    Optimize quaternion trajectory using `lax.scan()` for sequential updates.
    """

    T = omega.shape[0]
    q0 = jnp.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion

    def body_fun(q_prev, inputs):
        omega_t, dt_t = inputs
        delta_q = exp_map_quaternion(jnp.insert(omega_t * dt_t / 2, 0, 0))
        q_next = quaternion_multiply(q_prev, delta_q)
        return q_next, q_next

    _, q_traj = jax.lax.scan(body_fun, q0, (omega[:-1], dt))

    q_traj = jnp.vstack([q0, q_traj])  # Add initial quaternion

    # Function to perform one iteration of gradient descent
    def loop_body(i, q_traj):
        grad_q = grad(vectorized_cost_function)(q_traj, omega, accel_data, dt)
        q_traj = q_traj - lr * grad_q
        q_traj = q_traj / (jnp.linalg.norm(q_traj, axis=1, keepdims=True) + 1e-8)  # Normalize

        # Compute and print cost function value
        cost_value = vectorized_cost_function(q_traj, omega, accel_data, dt)
        jax.debug.print("Iteration {}: Cost = {}", i + 1, cost_value)

        return q_traj

    # Use `lax.fori_loop()` to iterate `num_iterations` times inside JIT
    q_traj = lax.fori_loop(0, num_iterations, loop_body, q_traj)

    return q_traj

def vectorized_motion_model(q_traj, omega, dt):
    """
    Vectorized motion model to update quaternion trajectory based on angular velocity.

    Parameters:
        q_traj (jnp.ndarray): Quaternion trajectory (Nx4).
        omega (jnp.ndarray): Angular velocity data (Nx3).
        dt (jnp.ndarray): Time step data (N-1,).

    Returns:
        jnp.ndarray: Updated quaternion trajectory (Nx4).
    """
    # Compute small rotation vectors
    axis = omega * dt[:, None] / 2

    delta_q = exp_map_quaternion_batch(jnp.insert(axis, 0, 0, axis=1))

    # Update quaternion trajectory
    q_pred = quaternion_multiply_batch(q_traj, delta_q)
    return q_pred / jnp.linalg.norm(q_pred, axis=-1, keepdims=True)  # Normalize



def quaternion_multiply_batch(q1, q2):
    """
    Perform quaternion multiplication for batches of quaternions using the following definition:

    q1 * q2 = [q1_s * q2_s - q1_v . q2_v, q1_s * q2_v + q2_s * q1_v + q1_v x q2_v]

    Parameters:
        q1 (jnp.ndarray): First batch of quaternions (Nx4).
        q2 (jnp.ndarray): Second batch of quaternions (Nx4).

    Returns:
        jnp.ndarray: Resulting batch of quaternions (Nx4).
    """
    q1_s, q1_v = q1[:, 0], q1[:, 1:]
    q2_s, q2_v = q2[:, 0], q2[:, 1:]

    scalar_part = q1_s * q2_s - jnp.sum(q1_v * q2_v, axis=-1)
    vector_part = (
        q1_s[:, None] * q2_v +
        q2_s[:, None] * q1_v +
        jnp.cross(q1_v, q2_v)
    )

    return jnp.concatenate([scalar_part[:, None], vector_part], axis=-1)

def quaternion_inverse_batch(q):
    """
    Compute the inverse of a batch of quaternions.

    Parameters:
        q (jnp.ndarray): Batch of quaternions (Nx4).

    Returns:
        jnp.ndarray: Inverse batch of quaternions (Nx4).
    """
    q_conj = jnp.concatenate([q[:, :1], -q[:, 1:]], axis=-1)  # Conjugate of quaternions
    q_norm_sq = jnp.sum(q ** 2, axis=-1, keepdims=True)        # Squared norm
    return q_conj / q_norm_sq  + 1e-8                               # Inverse formula


def exp_map_quaternion_batch(q_batch):
    """
    Compute the quaternion exponential map for a batch of quaternions.

    Parameters:
        q_batch (jnp.ndarray): Batch of quaternions (Nx4), where the first column is the scalar part (q_s),
                               and the last three columns are the vector part (q_v).

    Returns:
        jnp.ndarray: Transformed batch of quaternions (Nx4) using the exponential map.
    """
    q_s = q_batch[:, 0]  # Scalar part (N,)
    q_v = q_batch[:, 1:]  # Vector part (Nx3)

    norm = jnp.linalg.norm(q_v, axis=-1, keepdims=True)  # Norm of the vector part (Nx1)
    exp_qs = jnp.exp(q_s)  # Exponential of the scalar part (N,)

    t1 = jnp.multiply(exp_qs.reshape(-1, 1), jnp.cos(norm))  # First term (N,)
    t2 = exp_qs[:, None] * (q_v / (norm + 1e-8)) * jnp.sin(norm)  # Second term (Nx3)

    return jnp.concatenate((t1.reshape(-1, 1), t2), axis=1)

def log_map_quaternion_batch(q_batch):
    """
    Compute the quaternion logarithm map for a batch of quaternions.

    Parameters:
        q_batch (jnp.ndarray): Batch of quaternions (Nx4), where the first column is the scalar part (q_s),
                               and the last three columns are the vector part (q_v).

    Returns:
        jnp.ndarray: Logarithm map of the quaternion batch (Nx4).
    """
    q_s = q_batch[:, 0]  # Scalar part (N,)
    q_v = q_batch[:, 1:]  # Vector part (Nx3)

    norm_q = jnp.linalg.norm(q_batch, axis=-1, keepdims=True)  # Norm of quaternions (Nx1)
    norm_v = jnp.linalg.norm(q_v, axis=-1, keepdims=True)  # Norm of vector part (Nx1)

    t1 = jnp.log(norm_q)  # Log of quaternion norm (Nx1)
    temp = jnp.arccos(jnp.clip(q_s / (norm_q[:, 0] + 1e-8), -0.999999, 0.999999))
    t2 = (q_v / (norm_v + 1e-8)) 
    t2 = t2 * temp.reshape(-1, 1)

    return jnp.concatenate((t1.reshape(-1, 1), t2), axis=1)


def vectorized_cost_function(q_traj, omega, accel_data, dt, g=jnp.array([0, 0, 1])):
    """
    Optimized cost function with vectorized operations for efficiency.

    Parameters:
        q_traj (jnp.ndarray): Quaternion trajectory (Nx4).
        omega (jnp.ndarray): Angular velocity data (Nx3).
        accel_data (jnp.ndarray): Accelerometer data (Nx3).
        dt (jnp.ndarray): Time step data (N-1,).
        g (jnp.ndarray): Gravity vector in world frame (3,).

    Returns:
        float: Total cost.
    """

    ##########
    rel_rotation = quaternion_multiply_batch(quaternion_inverse_batch(q_traj[1:]), q_traj[:-1])
    motion_error = jnp.sum(jnp.linalg.norm(2*log_map_quaternion_batch(rel_rotation), axis=-1) ** 2)

    # Observation model predictions
    g_quat = jnp.array([0, *g])

    q_inv = quaternion_inverse_batch(q_traj)

    rotated_g = quaternion_multiply_batch(
        quaternion_multiply_batch(q_inv, g_quat[None,:]),
        q_traj
    )

    accel_world = rotated_g
    accel_data = jnp.insert(accel_data, 0, 0, axis=1)

    # Observation model error
    observation_error = jnp.sum(jnp.linalg.norm(accel_data - accel_world, axis=-1) ** 2)

    return 0.5 * (motion_error + observation_error)
    # return 0.5 * motion_error
    # return 0.5 * observation_error


'''

Debugging functions

'''

def check_observation_model_gravity(q_traj, accel_data, g=jnp.array([0, 0, 1])):
    """
    Check if the observation model gravity matches the ground truth gravity.

    Parameters:
        q_traj (jnp.ndarray): Quaternion trajectory (Nx4).
        accel_data (jnp.ndarray): Accelerometer data (Nx3).
        g (jnp.ndarray): Ground truth gravity vector in world frame (3,).

    Returns:
        None
    """
    # Compute gravity predictions in the IMU frame
    g_quat = jnp.array([0, *g])  # Quaternion representation of gravity
    q_inv = quaternion_inverse_batch(q_traj)
    print(q_inv)
    rotated_g = quaternion_multiply_batch(
        quaternion_multiply_batch(q_inv, g_quat[None, :]), q_traj
    )
    predicted_accel = rotated_g[:, 1:]

    # Compute the difference between predicted and measured acceleration
    diff = accel_data - predicted_accel

    mean_diff = jnp.mean(jnp.linalg.norm(diff, axis=1))

    print(f"Mean difference between predicted gravity and accelerometer readings: {mean_diff:.6f}")

def check_motion_model_cost(q_traj, omega, dt):
    """
    Check if the motion model cost is initialized to 0 for the first iteration.

    Parameters:
        q_traj (jnp.ndarray): Quaternion trajectory (Nx4).
        omega (jnp.ndarray): Angular velocity data (Nx3).
        dt (jnp.ndarray): Time step data (N-1,).

    Returns:
        None
    """
    T = omega.shape[0]
    q0 = jnp.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion

    def body_fun(q_prev, inputs):
        omega_t, dt_t = inputs
        delta_q = exp_map_quaternion(jnp.insert(omega_t * dt_t / 2, 0, 0))
        q_next = quaternion_multiply(q_prev, delta_q)
        return q_next, q_next

    _, q_traj = jax.lax.scan(body_fun, q0, (omega[:-1], dt))

    q_traj = jnp.vstack([q0, q_traj])  # Add initial quaternion


    rel_rotation = quaternion_multiply_batch(quaternion_inverse_batch(q_traj[1:]), q_traj[:-1])
    motion_error = jnp.sum(jnp.linalg.norm(2 * log_map_quaternion_batch(rel_rotation), axis=-1) ** 2)

    print(f"Initial motion model cost: {motion_error:.6f}")


