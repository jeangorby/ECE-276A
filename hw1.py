import jax
import jax.numpy as jnp

'''
f(q)
'''

def f(q):
    return jnp.log(jnp.linalg.norm(q))

grad_f = jax.grad(f)

q = jnp.array([1.0, 2.0, 3.0, 4.0])
gradient = grad_f(q)
print("Numerical Gradient:", gradient)

def analytical_gradient(q):
    norm_q = jnp.linalg.norm(q)
    return q / (norm_q ** 2)

analytical_grad = analytical_gradient(q)
print("Analytical Gradient:", analytical_grad)

'''
g(q)
'''

def g(q):
    q_v = q[1:]  # q_v = [q2, q3, q4]
    return q_v / jnp.linalg.norm(q_v)

jac_g = jax.jacrev(g)

q = jnp.array([1.0, 2.0, 3.0, 4.0])
jacobian_g = jac_g(q)
print("Numerical Jacobian of g:", jacobian_g)

def analytical_jacobian_g(q):
    q_v = q[1:]
    norm_qv = jnp.linalg.norm(q_v)
    eye = jnp.eye(3)  # Identity matrix for q_v
    outer = jnp.outer(q_v, q_v)
    return jnp.vstack([jnp.zeros(3), (eye / norm_qv - outer / norm_qv**3)]).transpose()

analytical_jac_g = analytical_jacobian_g(q)
print("Analytical Jacobian of g:", analytical_jac_g)

'''
h(q)
'''

def h(q):
    q_s = q[0]  # q_s = q1
    norm_q = jnp.linalg.norm(q)
    return jnp.arccos(q_s / norm_q)

grad_h = jax.grad(h)

gradient_h = grad_h(q)
print("Numerical Gradient of h:", gradient_h)

def analytical_gradient_h(q):
    q_s = q[0]
    norm_q = jnp.linalg.norm(q)
    term1 = -1 / jnp.sqrt(1 - (q_s / norm_q) ** 2)
    grad = term1 * (jnp.array([1, 0, 0, 0]) / norm_q - q_s * q / norm_q**3)
    return grad

analytical_grad_h = analytical_gradient_h(q)
print("Analytical Gradient of h:", analytical_grad_h)

'''
3a
'''

import numpy as np

R = np.array([
    [0.1729, -0.1468,  0.9739],
    [0.9739,  0.1729, -0.1468],
    [-0.1468,  0.9739,  0.1729]
])

trace_R = np.trace(R)
theta = np.arccos((trace_R - 1) / 2)

alpha = (1 / (2 * np.sin(theta))) * np.array([
    R[2, 1] - R[1, 2],
    R[0, 2] - R[2, 0],
    R[1, 0] - R[0, 1]
])

# Output the results
print("Rotation Angle (theta):", theta)
print("Rotation Axis (alpha):", alpha)

