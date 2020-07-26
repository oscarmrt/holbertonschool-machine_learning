This project is about 0x02. Calculus
0-sigma_is_for_sum - \sum_{i=2}^{5} i
3 + 4 + 5
3 + 4
2 + 3 + 4 + 5
2 + 3 + 4

1-seegma - \sum_{k=1}^{4} 9i - 2k
90 - 20
36i - 20
90 - 8k
36i - 8k

2-pi_is_for_product - \prod_{i = 1}^{m} i
(m - 1)!
0
(m + 1)!
m!

3-pee - \prod_{i = 0}^{10} i
10!
9!
100
0

4-hello_derivatives - \frac{dy}{dx} where y = x^4 + 3x^3 - 5x + 1
3x^3 + 6x^2 -4
4x^3 + 6x^2 - 5
4x^3 + 9x^2 - 5
4x^3 + 9x^2 - 4

5-log_on_fire - \frac{d (xln(x))}{dx}
ln(x)
\frac{1}{x} + 1
ln(x) + 1
\frac{1}{x}

6-voltaire - \frac{d (ln(x^2))}{dx}
\frac{2}{x}
\frac{1}{x^2}
\frac{2}{x^2}
\frac{1}{x}

7-partial_truths - \frac{\partial f(x, y)}{\partial y} where f(x, y) = e^{xy} and  \frac{\partial&space;x}{\partial&space;y}=\frac{\partial&space;y}{\partial&space;x}=0
e^{xy}
ye^{xy}
xe^{xy}
e^{x}

8-all-together - \frac{\partial^2}{\partial&space;y\partial&space;x}(e^{x^2y}) where \frac{\partial&space;x}{\partial&space;y}=\frac{\partial&space;y}{\partial&space;x}=0
2x(1+y)e^{x^2y}
2xe^{2x}
2x(1+x^2y)e^{x^2y}
e^{2x}

9-sum_total.py - Write a function def summation_i_squared(n): that calculates \sum_{i=1}^{n} i^2:
n is the stopping condition
Return the integer value of the sum
If n is not a valid number, return None
You are not allowed to use any loops

10-matisse.py - Write a function def poly_derivative(poly): that calculates the derivative of a polynomial:
poly is a list of coefficients representing a polynomial
the index of the list represents the power of x that the coefficient belongs to
Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
If poly is not valid, return None
If the derivative is 0, return [0]
Return a new list of coefficients representing the derivative of the polynomial

11-integral - 
3x2 + C
x4/4 + C
x4 + C
x4/3 + C

12-integral - 
e2y + C
ey + C
e2y/2 + C
ey/2 + C

13-definite - 
3
6
9
27

14-definite - 
-1
0
1
undefined

15-definite - 
5
5x
25
25x

16-double - 
9ln(2)
9
27ln(2)
27

17-integrate.py - Write a function def poly_integral(poly, C=0): that calculates the integral of a polynomial:
poly is a list of coefficients representing a polynomial
the index of the list represents the power of x that the coefficient belongs to
Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
C is an integer representing the integration constant
If a coefficient is a whole number, it should be represented as an integer
If poly or C are not valid, return None
Return a new list of coefficients representing the integral of the polynomial
The returned list should be as small as possible
