import sympy as sp

x, y, z = sp.symbols("x y z", real=True)
m, n, k = sp.symbols("m n k", integer=True)
pi = sp.pi

print("Searching for solution:")
print("  phi(x,y,z) = sin(m*pi*x) * sin(n*pi*y) * sin(k*pi*z)")
print("  psi + Laplacian(phi) - (phi^2 - 1)*phi = 0")
print("  phi = 0 and psi = 0 on the whole boundary\n")

# Define function phi
phi = sp.sin(m*pi*x) * sp.sin(n*pi*y) * sp.sin(k*pi*z)

# Compute Laplacian of phi
laplace_phi = sp.diff(phi, x, 2) + sp.diff(phi, y, 2) + sp.diff(phi, z, 2)
laplace_phi = sp.simplify(laplace_phi)

# Find psi from second equation: psi = (phi^2 - 1)*phi - laplace(phi)
psi = sp.expand((phi**2 - 1)*phi - laplace_phi)

# Compute f = laplace(psi)
laplace_psi = sp.diff(psi, x, 2) + sp.diff(psi, y, 2) + sp.diff(psi, z, 2)
f = sp.expand(laplace_psi)


def to_cpp(expr, indent="    "):
    """Convert sympy expression to C++ code with std:: functions."""
    code = sp.ccode(expr)
    # Replace C functions with std:: versions
    code = code.replace("pow(", "std::pow(")
    code = code.replace("sin(", "std::sin(")
    code = code.replace("cos(", "std::cos(")
    # Replace M_PI with PI constant
    code = code.replace("M_PI", "PI")
    # Split long lines
    lines = []
    current = ""
    terms = code.replace(" - ", " + -").split(" + ")
    for i, term in enumerate(terms):
        term = term.strip()
        if not term:
            continue
        if i == 0:
            if term.startswith("-"):
                current = term
            else:
                current = term
        elif len(current) + len(term) > 80:
            if term.startswith("-"):
                lines.append(current + " -")
                current = indent + term[1:]
            else:
                lines.append(current + " +")
                current = indent + term
        else:
            if term.startswith("-"):
                current += " - " + term[1:]
            else:
                current += " + " + term
    if current:
        lines.append(current)
    return "\n".join(lines)


print("=== GENERATED EXPRESSIONS (C/C++) ===\n")

print("phi(x,y,z) =")
print(to_cpp(phi))
print()

print("psi(x,y,z) =")
print(to_cpp(psi))
print()

print("f(x,y,z) =")
print(to_cpp(f))
print()
