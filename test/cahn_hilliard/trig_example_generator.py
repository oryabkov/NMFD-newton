import sympy as sp
import numpy as np

x, y, z = sp.symbols("x y z", real=True)
pi = sp.pi

print("Searching for solution:")
print("  phi(x,y,z) = sin(pi*x) * sin(pi*y) * sin(pi*z)")
print("  psi + Laplacian(phi) - (phi^2 - 1)*phi = 0")
print("  phi = 0 and psi = 0 on the whole boundary\n")

# Define function phi
phi = sp.sin(pi*x) * sp.sin(pi*y) * sp.sin(pi*z)

print("1. Function phi:")
print(f"phi = {phi}")
print()

# Compute Laplacian of phi
laplace_phi = sp.diff(phi, x, 2) + sp.diff(phi, y, 2) + sp.diff(phi, z, 2)
laplace_phi_simplified = sp.simplify(laplace_phi)

print("2. Laplacian of phi (∇²φ):")
print(f"∇²φ = {laplace_phi_simplified}")
print()

# Find psi from second equation
# ψ + ∇²φ - (φ² - 1)φ = 0 => ψ = (φ² - 1)φ - ∇²φ
psi = (phi**2 - 1)*phi - laplace_phi_simplified
psi_simplified = sp.simplify(psi)

print("3. Function ψ from second equation:")
print(f"ψ = {psi_simplified}")
print()

# Check boundary conditions for psi
print("4. Boundary condition check for ψ:")
# Check at points x=0, x=1
psi_at_x0 = psi_simplified.subs(x, 0)
psi_at_x1 = psi_simplified.subs(x, 1)
print(f"ψ(0, y, z) = {sp.simplify(psi_at_x0)}")
print(f"ψ(1, y, z) = {sp.simplify(psi_at_x1)}")
print()

# Compute Laplacian of psi
laplace_psi = sp.diff(psi_simplified, x, 2) + sp.diff(psi_simplified, y, 2) + sp.diff(psi_simplified, z, 2)
laplace_psi_simplified = sp.simplify(laplace_psi)

print("5. Laplacian of ψ (∇²ψ):")
print(f"∇²ψ = {laplace_psi_simplified}")
print()

# Find f from first equation: ∇²ψ - f = 0 => f = ∇²ψ
f = laplace_psi_simplified

print("6. Function f from first equation:")
print(f"f = {f}")
print()

# Simplify expression for f by expanding and grouping
f_expanded = sp.expand(f)
f_grouped = sp.collect(f_expanded, [sp.sin(pi*x), sp.sin(pi*y), sp.sin(pi*z)])

print("7. Simplified expression for f:")
print(f"f = {f_grouped}")
print()

# Check some values at random points for internal consistency
print("8. Verification at random point (x=0.3, y=0.4, z=0.5):")
test_point = {x: 0.3, y: 0.4, z: 0.5}
phi_val = float(phi.subs(test_point))
psi_val = float(psi_simplified.subs(test_point))
f_val = float(f.subs(test_point))

# Check equation satisfaction
laplace_phi_val = float(laplace_phi_simplified.subs(test_point))
# Equation 2: ψ + ∇²φ - (φ² - 1)φ should be 0
eq2_check = psi_val + laplace_phi_val - (phi_val**2 - 1)*phi_val
print(f"φ = {phi_val:.6f}")
print(f"ψ = {psi_val:.6f}")
print(f"∇²φ = {laplace_phi_val:.6f}")
print(f"Check of second equation: ψ + ∇²φ - (φ²-1)φ = {eq2_check:.2e}")

# For checking first equation, need to compute ∇²ψ numerically
# But better to do it symbolically
laplace_psi_val = float(laplace_psi_simplified.subs(test_point))
eq1_check = laplace_psi_val - f_val
print(f"∇²ψ = {laplace_psi_val:.6f}")
print(f"f = {f_val:.6f}")
print(f"Check of first equation: ∇²ψ - f = {eq1_check:.2e}")
print()

# Output compact formulas
print("9. Final formulas:")
print(f"φ(x, y, z) = sin(πx)·sin(πy)·sin(πz)")
print(f"∇²φ = -3π²·sin(πx)·sin(πy)·sin(πz)")
print()
print("ψ = (φ² - 1)φ - ∇²φ")
print(f"  = [sin²(πx)·sin²(πy)·sin²(πz) - 1 + 3π²]·sin(πx)·sin(πy)·sin(πz)")
print()
print("f = ∇²ψ")
print(f"  = {sp.latex(f)}")

# Additionally: compare with previous manual calculations
print("\n" + "="*80)
print("COMPARISON WITH MANUAL CALCULATIONS:")
print("="*80)

# Previous version of psi
psi_my_version = (3*pi**2 - 1 + phi**2) * phi
print(f"\nMy version of ψ: {psi_my_version}")
print(f"SymPy version of ψ: {psi_simplified}")
print(f"Difference: {sp.simplify(psi_my_version - psi_simplified)}")

# Previous version of f (simplified form)
# Use simplified expression from SymPy
f_sympy_simplest = sp.simplify(f)
print(f"\nSymPy f (maximally simplified):")
print(f"f = {f_sympy_simplest}")
print(f"Expression length: {len(str(f_sympy_simplest))} characters")
