import numpy as np

'''

Question 1

'''


# Define zi as columns (each column is a point in 3D space)
zi = np.array([
    [0.2785, 0.5469, 0.9575],  # Point 1
    [0.1576, 0.9706, 0.9572],  # Point 2
    [0.8003, 0.1419, 0.4218],  # Point 3
    [0.7922, 0.9595, 0.6557],  # Point 4
    [0.8491, 0.9340, 0.6787],  # Point 5
    [0.7431, 0.3922, 0.6555],  # Point 6
    [0.7060, 0.0318, 0.2769],  # Point 7
    [0.0971, 0.8235, 0.6948]   # Point 8
]).T

# Define mi as columns (each column is a point in 3D space)
mi = np.array([
    [0.5391, -0.5638, 1.8256],  # Point 1
    [0.2632, -0.7963, 1.5871],  # Point 2
    [0.4950, 0.2746, 1.6287],   # Point 3
    [0.2118, -0.1363, 1.4197],  # Point 4
    [0.2699, -0.1258, 2.1640],  # Point 5
    [0.5128, 0.0386, 1.1055],   # Point 6
    [0.4039, 0.3606, 0.5774],   # Point 7
    [0.1199, -0.6349, 1.1945]   # Point 8
]).T 

wi = np.array([1, 1, 1, 1, 1, 1, 1, 1])

# Compute weighted centroids
w_sum = np.sum(wi)

m_bar = np.sum(wi * mi, axis=1) / w_sum  # Weighted mean of mi
z_bar = np.sum(wi * zi, axis=1) / w_sum  # Weighted mean of zi

# Compute deviations from centroids
delta_mi = mi - m_bar[:, np.newaxis]  # Each column is mi - m_bar
delta_zi = zi - z_bar[:, np.newaxis]  # Each column is zi - z_bar

# Compute Q matrix
Q = np.dot(delta_mi * wi[np.newaxis, :], delta_zi.T)

# Perform SVD
U, Sigma, Vt = np.linalg.svd(Q)

det_term = np.linalg.det(np.dot(U, Vt))  # Compute det(UV^T)
D = np.eye(3)  # Identity matrix (3x3)
D[2, 2] = det_term  # Modify the last diagonal entry

# Compute R
R = np.dot(U, np.dot(D, Vt))

# Compute translation vector ρ
p = m_bar - np.dot(R, z_bar)

print("Rotation Matrix: ")
print(R)

print("Translation: ")
print(p)

# test = np.dot(R, zi[:,1]) + p

# print(test)
# print(mi[:,1])

print(np.linalg.det(R))

'''

Question 2

'''

# Define z1i as columns (each column is a point in homogeneous 3D space)
z1i = np.array([
    [0.2953, -0.3088, 1],  # Point 1
    [0.1658, -0.5017, 1],  # Point 2
    [0.3040, 0.1686, 1],   # Point 3
    [0.1492, -0.0960, 1],  # Point 4
    [0.1247, -0.0581, 1],  # Point 5
    [0.4638, 0.0349, 1],   # Point 6
    [0.6996, 0.6246, 1],   # Point 7
    [0.1004, -0.5315, 1]   # Point 8
]).T  

# Define z2i as columns (each column is a point in homogeneous 3D space)
z2i = np.array([
    [0.2909, 0.5712, 1],   # Point 1
    [0.1647, 1.0140, 1],   # Point 2
    [1.8975, 0.3364, 1],   # Point 3
    [1.2081, 1.4632, 1],   # Point 4
    [1.2510, 1.3761, 1],   # Point 5
    [1.1337, 0.5984, 1],   # Point 6
    [2.5496, 0.1150, 1],   # Point 7
    [0.1398, 1.1851, 1]    # Point 8
]).T 

k1 = np.eye(3)
k2 = np.eye(3)

y1i = np.linalg.inv(k1) @ z1i  # y1i = K1^-1 * z1i
y2i = np.linalg.inv(k2) @ z2i  # y2i = K2^-1 * z2i

yi_vecs = []

# Loop over each column of y2i
for i in range(y2i.shape[1]):

    outer_product = np.multiply(y1i[:, i].reshape((3,1)), y2i[:, i].reshape((3,1)).T)  # Compute product

    vectorized = np.ravel(outer_product, order='F')  # Flatten to a vector

    yi_vecs.append(vectorized)  # Append to the list

# Stack the column vectors into a matrix
y_bar = np.array(yi_vecs)

U, S, Vt = np.linalg.svd(y_bar) 

e = Vt[-1, :]  # Last row of V^T (i.e., last column of V)

E = e.reshape(3, 3).T

# Compute the SVD decomposition of E
U, Sigma, Vt = np.linalg.svd(E)

Sigma = np.diag(Sigma)

# To ensure the determinant is +1
Vt[-1, :] *= -1  
Sigma[-1, -1] *= -1  

# Define the 90-degree rotation matrix Rz(pi/2) and Rz(-pi/2)
Rz_pos = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]
])  

Rz_neg = np.array([
    [0,  1, 0],
    [-1, 0, 0],
    [0,  0, 1]
])  

# Compute the two possible solutions for (p_hat, R)
P_hat_1 = U @ Rz_pos @ Sigma @ U.T
R1 = U @ Rz_pos.T @ Vt

P_hat_2 = U @ Rz_neg @ Sigma @ U.T
R2 = U @ Rz_neg.T @ Vt

def vee_map(p_hat):
    """
    Recovers the vector p from its skew-symmetric matrix representation p_hat.
    """
    p1 = p_hat[2, 1]  # Extract p1 from (3,2) position
    p2 = p_hat[0, 2]  # Extract p2 from (1,3) position
    p3 = p_hat[1, 0]  # Extract p3 from (2,1) position
    
    return np.array([p1, p2, p3])


p1 = vee_map(P_hat_1)
p2 = vee_map(P_hat_2)

print("")
print("Guess 1:")
print("Translation: ")
print(p1)

print("Rotation Matrix: ")
print(R1)

print("")

print("Guess 2:")
print("Translation: ")
print(p2)

print("Rotation Matrix: ")
print(R2)

'''

question 1

'''

def rotation_matrix_R(t):
    """ Computes the rotation matrix R(t) """
    R = np.array([
        [np.cos(t * np.pi / 3),  0, -np.sin(t * np.pi / 3)],
        [0,                      1,  0],
        [np.sin(t * np.pi / 3),  0,  np.cos(t * np.pi / 3)]
    ])
    return R

def derivative_rotation_matrix_Rdot(t):
    """ Computes the time derivative of R(t), i.e., Ṙ(t) """
    R_dot = np.array([
        [-np.pi / 3 * np.sin(t * np.pi / 3),  0, -np.pi / 3 * np.cos(t * np.pi / 3)],
        [0,                                   0,  0],
        [ np.pi / 3 * np.cos(t * np.pi / 3),  0, -np.pi / 3 * np.sin(t * np.pi / 3)]
    ])
    return R_dot

def compute_twist_xi(t, p_t, p_dot_t):
    """ Computes the body-frame twist xi_hat(t) """

    # Compute R(t) and Ṙ(t)
    R_t = rotation_matrix_R(t)
    R_dot_t = derivative_rotation_matrix_Rdot(t)

    # Compute R^T(t)
    R_T_t = R_t.T

    # Compute the body-frame velocity components
    xi_rotational = R_T_t @ R_dot_t  # R^T * Ṙ
    xi_translational = -R_T_t @ p_t  # -R^T * p

    # Construct xi_hat
    xi_hat = np.block([
        [xi_rotational, xi_translational.reshape(3,1)],
        [np.zeros((1,3)), np.array([[1]])]
    ])

    return xi_hat

# Define time t = 1
t = 1

# Define p(t) and its derivative ṗ(t)
p_t = np.array([t, 0, 2*t])  # p(t) = [t, 0, 2t]^T
p_dot_t = np.array([1, 0, 2])  # ṗ(t) = [1, 0, 2]^T

# Compute xi_hat at t = 1
xi_hat_t = compute_twist_xi(t, p_t, p_dot_t)

# Print results
print("ξ̂(t) at t = 1:\n", xi_hat_t)