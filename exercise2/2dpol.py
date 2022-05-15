from numpy import polynomial
import numpy as np

x = [0,1,2]
y = [2,2,2]
c = np.asanyarray([[0,0],[1,1]])

print(polynomial.polynomial.polyval2d(x,y,c))
