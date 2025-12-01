import numpy as np


def solveAXEqualsZero(A):
    # TO DO: Write this routine - it should solve Ah = 0. You can do this using SVD. Check the task above.
    # Hint: SVD will be involved.
    U, S, Vt = np.linalg.svd(A)

    
    return U, S, Vt


def svd_from_scratch(A, epsilon=1e-10, max_iterations=100):
    """
    Computes SVD using Power Iteration with Deflation.
    Returns: U, Sigma, Vt
    """
    A = A.astype(float)
    m, n = A.shape
    
    # We can perform SVD up to the smallest dimension of A
    num_components = min(m, n)
    
    # Initialize output matrices
    U = np.zeros((m, num_components))
    S = np.zeros(num_components)
    Vt = np.zeros((num_components, n))
    
    # Create a working copy of A so we don't modify the original during deflation
    A_working = A.copy()

    for i in range(num_components):
        # --- Step 1: Power Iteration ---
        # Start with a random vector v (right singular vector)
        v = np.random.rand(n)
        v = v / np.linalg.norm(v) # Normalize
        
        u = np.zeros(m)
        sigma = 0
        
        for _ in range(max_iterations):
            # 1. Map v to u space: u = A * v
            # Note: We use the current working matrix (A_working)
            u_unscaled = np.dot(A_working, v)
            
            # 2. Update sigma (singular value is the magnitude of A*v)
            sigma = np.linalg.norm(u_unscaled)
            
            # Avoid division by zero
            if sigma < epsilon:
                u = np.zeros(m)
                break
                
            # 3. Normalize u
            u = u_unscaled / sigma
            
            # 4. Map u back to v space: v = A.T * u
            v_unscaled = np.dot(A_working.T, u)
            
            # 5. Normalize v
            v = v_unscaled / np.linalg.norm(v_unscaled)
        
        # --- Step 2: Store Results ---
        S[i] = sigma
        U[:, i] = u
        Vt[i, :] = v # Vt stores v as rows
        
        # --- Step 3: Deflation ---
        # Remove this singular component from the matrix
        # A_new = A - sigma * (u * v.T)
        # We use outer product to create the matrix to subtract
        component = sigma * np.outer(u, v)
        A_working = A_working - component

    return U, S, Vt


# compare the two SVD implementations
A = np.random.rand(5, 3)
U1, S1, Vt1 = np.linalg.svd(A, full_matrices=False)
U2, S2, Vt2 = svd_from_scratch(A)
print("NumPy SVD Results:")
print("U:\n", U1)
print("S:\n", S1)
print("Vt:\n", Vt1)
print("\nCustom SVD Results:")
print("U:\n", U2)
print("S:\n", S2)
print("Vt:\n", Vt2)

# Verify reconstruction
print("\nReconstruction Error:")
A_reconstructed_numpy = U1 @ np.diag(S1) @ Vt1
error_numpy = np.linalg.norm(A - A_reconstructed_numpy)
print(f"NumPy reconstruction error: {error_numpy:.2e}")

A_reconstructed_custom = U2 @ np.diag(S2) @ Vt2
error_custom = np.linalg.norm(A - A_reconstructed_custom)
print(f"Custom reconstruction error: {error_custom:.2e}")

# Check if they are equivalent up to sign
print("\nChecking sign flips:")
for i in range(len(S1)):
    # Check dot product of columns of U
    dot_u = np.dot(U1[:, i], U2[:, i])
    # Check dot product of rows of Vt
    dot_v = np.dot(Vt1[i, :], Vt2[i, :])
    
    print(f"Component {i}: U dot product = {dot_u:.2f}, Vt dot product = {dot_v:.2f}")
    if np.isclose(abs(dot_u), 1.0) and np.isclose(abs(dot_v), 1.0):
        if np.sign(dot_u) == np.sign(dot_v):
             print(f"  -> Component {i} is consistent (signs match or both flipped)")
        else:
             print(f"  -> Component {i} is INCONSISTENT (signs mismatch)")
    else:
        print(f"  -> Component {i} vectors do not match in magnitude/direction")

