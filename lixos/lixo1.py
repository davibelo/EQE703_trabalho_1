# Sphere functions
def f1(r):
    return (r[0] - r1[0])**2 + (r[1] - r1[1])**2 + (r[2] - r1[2])**2 - 100

def f2(r):
    return (r[0] - r2[0])**2 + (r[1] - r2[1])**2 + (r[2] - r2[2])**2 - 100

def f3(r):
    return (r[0] - r3[0])**2 + (r[1] - r3[1])**2 + (r[2] - r3[2])**2 - 100

# Objetive Funcion G(r)
def G(r):
    return f1(r)**2 + f2(r)**2 + f3(r)**2

# Initial r
r_init = np.array([0, 0, 0])

# Objetive funcion on initial r
G_value = G(r_init)
print("Objetive function value on initial r:", G_value)

# Gradient functions of spheres
def grad_f1(r):
    return np.array([2*(r[0] - r1[0]), 2*(r[1] - r1[1]), 2*(r[2]-r1[2])])

def grad_f2(r):
    return np.array([2*(r[0] - r2[0]), 2*(r[1] - r2[1]), 2*(r[2]-r2[2])])

def grad_f3(r):
    return np.array([2*(r[0] - r3[0]), 2*(r[1] - r2[1]), 2*(r[2]-r2[2])])

# Gradient of objective function 
def grad_G(r):
    return 2 * f1(r) * grad_f1(r) + 2 * f2(r) * grad_f2(r) + 2 * f3(r) * grad_f3(r)

# Hessian Objective function
def hessian_G(r):
    H = np.zeros((3, 3))
    
    grad_f = [grad_f1, grad_f2, grad_f3]
    f = [f1, f2, f3]
    
    for i in range(3):
        grad_fi = grad_f[i](r)
        fi = f[i](r)
        H += 2 * (np.outer(grad_fi, grad_fi) + fi * hessian_fi(r, grad_f[i]))
    
    return H

# Hessian of f_i
def hessian_fi(r, grad_fi):
    hessian_fi = np.zeros((3, 3))
    if grad_fi == grad_f1:
        hessian_fi = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    elif grad_fi == grad_f2:
        hessian_fi = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    elif grad_fi == grad_f3:
        hessian_fi = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    return hessian_fi

# Cálculo do gradiente e da matriz Hessiana no ponto inicial
grad_G_value = grad_G(r_init)
hessian_G_value = hessian_G(r_init)

print("Objective function gradient on initial r:", grad_G_value)
print("Hessian matrix of objective function on initial r:\n", hessian_G_value)