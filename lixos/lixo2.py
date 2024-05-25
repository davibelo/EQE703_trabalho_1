# Define a wrapper function for optimization
def F_Z_wrapper(Z):
    alpha, beta = Z
    return F_Z(alpha, beta)

# Initial guess for alpha and beta
initial_guess = [0, 0]

# Perform the optimization
result = scipy.optimize.minimize(F_Z_wrapper, initial_guess, method='BFGS')

# Extract the optimal alpha and beta
opt_alpha, opt_beta = result.x

# Print the results
print(f"Optimal alpha: {opt_alpha}")
print(f"Optimal beta: {opt_beta}")
print(f"Minimum F_Z value: {F_Z(opt_alpha, opt_beta)}")

# Create a meshgrid for alpha and beta for plotting
alpha_vals = np.linspace(opt_alpha-1e5, opt_alpha+1e5, 100)
beta_vals = np.linspace(opt_beta-1e5, opt_beta+1e5, 100)
Alpha, Beta = np.meshgrid(alpha_vals, beta_vals)

# Compute F(X) for each pair (alpha, beta)
F_vals = np.vectorize(F_Z)(Alpha, Beta)

# Plotting
plt.figure(figsize=(10, 7))
contour = plt.contourf(Alpha, Beta, F_vals, levels=50, cmap='coolwarm')
plt.colorbar(contour)
plt.plot(opt_alpha, opt_beta, 'ro')  # Mark the optimal point
plt.title('Contour plot of F(X)')
plt.xlabel('Alpha')
plt.ylabel('Beta')
plt.show()


opt_Z = np.array([opt_alpha, opt_beta])

# finding x1 to x6 with alpha, beta values
opt_X_red_byZ = np.dot(K, opt_Z)

# appending alpha, beta on X_red
opt_X_byZ = np.append(opt_X_red_byZ, [opt_alpha, opt_beta])

# Print the results
print(f"Optimal X: {opt_X_byZ}")
print(f"Minimum F_X value: {F_X(opt_X_byZ)}")
