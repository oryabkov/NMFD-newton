import sympy as sp
import numpy as np

x, y, z = sp.symbols("x y z")

P_DEGREE = 6

print("Searching for solution:")
print("  phi(x,y,z) = g(x,y,z) * P(x,y,z)")
print("  psi + Laplacian(phi) - (phi^2 - 1)*phi = 0")
print("  phi = 0 and psi = 0 on the whole boundary\n")

g = x*(1 - x)*y*(1 - y)*z*(1 - z)

coeffs = []
P = 0
idx = 0

for i in range(P_DEGREE + 1):
    for j in range(P_DEGREE + 1 - i):
        for k in range(P_DEGREE + 1 - i - j):
            a = sp.symbols(f"a{idx}")
            coeffs.append(a)
            P += a * x**i * y**j * z**k
            idx += 1

print(f"Polynomial degree        : {P_DEGREE}")
print(f"Number of coefficients   : {len(coeffs)}")

phi = g * P
lap_phi = sp.diff(phi, x, 2) + sp.diff(phi, y, 2) + sp.diff(phi, z, 2)

def collect_coeffs(expr, subs, vars):
    poly = sp.Poly(sp.expand(expr.subs(subs)), vars)
    return poly.coeffs()

equations = []
equations += collect_coeffs(lap_phi, {x: 0}, (y, z))
equations += collect_coeffs(lap_phi, {x: 1}, (y, z))
equations += collect_coeffs(lap_phi, {y: 0}, (x, z))
equations += collect_coeffs(lap_phi, {y: 1}, (x, z))
equations += collect_coeffs(lap_phi, {z: 0}, (x, y))
equations += collect_coeffs(lap_phi, {z: 1}, (x, y))

print(f"Number of equations      : {len(equations)}")

solution = sp.linsolve(equations, coeffs)
free = list(solution.free_symbols)

print(f"Solution space dimension : {len(free)}")

if not free:
    raise RuntimeError("No non-trivial solution")

sol = list(solution)[0]
sol = [s.subs(free[0], 1) for s in sol]

P = sp.expand(P.subs(dict(zip(coeffs, sol))))
phi = g * P

lap_phi = sp.diff(phi, x, 2) + sp.diff(phi, y, 2) + sp.diff(phi, z, 2)
psi = sp.expand((phi**2 - 1)*phi - lap_phi)

lap_psi = sp.diff(psi, x, 2) + sp.diff(psi, y, 2) + sp.diff(psi, z, 2)
f = sp.expand(lap_psi)

def group_by_degree(expr):
    poly = sp.Poly(expr, x, y, z)
    groups = {}
    for monom, coeff in poly.terms():
        deg = sum(monom)
        term = coeff
        for v, p in zip((x, y, z), monom):
            if p > 0:
                term *= v**p
        groups.setdefault(deg, []).append(term)
    return groups

def print_cpp_grouped(expr, indent="  "):
    groups = group_by_degree(expr)
    first = True
    for deg in sorted(groups):
        line = ""
        for t in groups[deg]:
            c = sp.Poly(t, x, y, z).coeff_monomial(
                x**sp.degree(t, x) *
                y**sp.degree(t, y) *
                z**sp.degree(t, z)
            )
            neg = c.could_extract_minus_sign()
            term = sp.ccode(-t if neg else t)
            if first:
                if neg:
                    line += "-"
                line += term
                first = False
            else:
                line += " - " + term if neg else " + " + term
        print(indent + line)

def try_factor_by_g(expr):
    r = sp.expand(sp.cancel(expr / g))
    if r.is_polynomial(x, y, z):
        return r
    return None

print("\n=== GENERATED EXPRESSIONS (C/C++) ===\n")

print("base factor")
print("g = x*(1-x)*y*(1-y)*z*(1-z);\n")

print("polynomial P")
print_cpp_grouped(P)
print()

print("phi(x,y,z) =")
print("g * (")
print_cpp_grouped(P)
print(")\n")

print("psi(x,y,z) =")
psi_r = try_factor_by_g(psi)
if psi_r is not None:
    print("g * (")
    print_cpp_grouped(psi_r)
    print(")\n")
else:
    print_cpp_grouped(psi)
    print()

print("f(x, y, z) =")
print_cpp_grouped(f)
print()

phi_f = sp.lambdify((x, y, z), phi, "numpy")
psi_f = sp.lambdify((x, y, z), psi, "numpy")

vals = np.linspace(0.0, 1.0, 6)
max_phi = 0.0
max_psi = 0.0

for X in vals:
    for Y in vals:
        for Z in vals:
            if X in (0.0, 1.0) or Y in (0.0, 1.0) or Z in (0.0, 1.0):
                max_phi = max(max_phi, abs(phi_f(X, Y, Z)))
                max_psi = max(max_psi, abs(psi_f(X, Y, Z)))

print("=== BOUNDARY VERIFICATION ===")
print(f"max |phi| on boundary = {max_phi:.3e}")
print(f"max |psi| on boundary = {max_psi:.3e}")
