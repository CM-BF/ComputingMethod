from sympy import *

x = symbols("x")
y = symbols("y")

f = x**2 + y**2 -1
g = x**3 - y
F = Matrix([[f], [g]])
e = 10**-5
X = [Matrix([[0.8], [0.6]])]

J = Matrix([[diff(f, x), diff(f, y)], [diff(g, x), diff(g, y)]])

k = 0
X.append(X[k] - J.evalf(subs={x:X[k][0], y:X[k][1]})**(-1) * F.evalf(subs={x:X[k][0], y:X[k][1]}))
k += 1
while abs(max(X[k][0] - X[k-1][0], X[k][1] - X[k-1][1])) >= e:
    X.append(X[k] - J.evalf(subs={x: X[k][0], y: X[k][1]}) ** (-1) * F.evalf(subs={x: X[k][0], y: X[k][1]}))
    k += 1


print(max(X[k][0] - X[k-1][0], X[k][1] - X[k-1][1]))
print(f.evalf(subs={x:X[k][0], y:X[k][1]}), g.evalf(subs={x:X[k][0], y:X[k][1]}))
print(k, X[k])
