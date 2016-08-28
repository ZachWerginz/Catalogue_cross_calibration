import numpy as np

class Measurement:
    #Set array priority to override some of ndarray's ufunc binary relations
    __array_priority__ = 10000

    def __init__(self, value, uncertainty):
        self.v = value
        self.u = uncertainty

    def __getitem__(self, index):
        return Measurement(self.v[index], self.u[index])

    def __setitem__(self, index, item):
        self.v[index] = item
        self.u[index] = item

    def __repr__(self):
        return 'Measurement({}, {})'.format(self.v, self.u)

    def __str__(self):
        return '{}+/-{}'.format(self.v, self.u)

    def v(self):
        return self.v

    def u(self):
        return self.u
    def __len__(self):
        return len(self.v)

    def __add__(self, right):
        if isinstance(right, (float, np.ndarray, int)):
            return Measurement(self.v + right, self.u)
        unc = np.sqrt(self.u**2 + right.u**2)
        return Measurement(self.v + right.v, unc)

    def __radd__(self, left):
        return Measurement(self.v + left, self.u)

    def __iadd__(self, other):
        result = self + other
        return result

    def __sub__(self, right):
        if isinstance(right, (float, np.ndarray, int)):
            return Measurement(self.v - right, self.u)
        unc = np.sqrt(self.u**2 + right.u**2)
        return Measurement(self.v - right.v, unc)

    def __rsub__(self, left):
        result = self*-1 + left
        return result

    def __mul__(self, right):
        if isinstance(right, (float, np.ndarray, int)):
            val = self.v*right
            unc = np.abs(self.u*right)
            return Measurement(val, unc)
        val = self.v*right.v
        unc = np.sqrt((self.u/self.v)**2 + (right.u/right.v)**2)
        return Measurement(val, np.abs(unc*val))

    def __rmul__(self, left):
        return Measurement(left*self.v, np.abs(left*self.u))

    def __truediv__(self, right):
        if isinstance(right, (float, np.ndarray, int)):
            return Measurement(self.v/right, np.abs(self.u/right))
        unc = np.sqrt((self.u/self.v)**2 + (right.u/right.v)**2)
        val = self.v/right.v
        return Measurement(val, np.abs(unc*val))

    def __rtruediv__(self, left):
        return left*self**(-1)

    def __pow__(self, power):
        val = self.v ** power
        unc = np.abs(val*power*self.u/self.v)
        return Measurement(val, unc)

    def __abs__(self):
        return Measurement(np.abs(self.v), self.u)

    def __gt__(a, b):
        try:
            return np.greater(a.v, b.v)
        except AttributeError:
            if isinstance(a, Measurement):
                return np.greater(a.v, b)
            else:
                return np.greater(a, b.v)

    def __ge__(a, b):
        try:
            return np.greater_equal(a.v, b.v)
        except AttributeError:
            if isinstance(a, Measurement):
                return np.greater_equal(a.v, b)
            else:
                return np.greater_equal(a, b.v)

    def __lt__(a, b):
        try:
            return np.less(a.v, b.v)
        except AttributeError:
            if isinstance(a, Measurement):
                return np.less(a.v, b)
            else:
                return np.less(a, b.v)
        
    def __and__(a, b):
        return np.logical_and(a, b)

    def __or__(a, b):
        return np.logical_or(a, b)

    def cos(x):
        try:
            return Measurement(np.cos(x.v), np.abs(np.sin(x.v)*x.u))
        except AttributeError:
            return np.cos(x)

    def sin(x):
        try:
            return Measurement(np.sin(x.v), np.abs(np.cos(x.v)*x.u))
        except AttributeError:
            return np.sin(x)

    def arctan(x):
        try:
            val = np.arctan(x.v)
            unc = 1/(1 + x.v*x.v)*x.u
            return Measurement(val, unc)
        except AttributeError:
            return np.arctan(x)

    def arctan2(x, y):
        try:
            val = np.arctan2(x.v, y.v)
            z = x/y
            unc = 1/(1 + z.v*z.v)*z.u
            return Measurement(val, np.abs(unc))
        except AttributeError:
            return np.arctan2(x, y)

    def arcsin(x):
        try:
            val = np.arcsin(x.v)
            unc = (1/np.sqrt(1 - x.v*x.v))*x.u
            return Measurement(val, np.abs(unc))
        except AttributeError:
            return np.arcsin(x)

    def sqrt(x):
        try:
            return Measurement(np.sqrt(x.v), 0.5*np.abs(x.u/np.sqrt(x.v)))
        except AttributeError:
            return np.sqrt(x)

    def deg2rad(x):
        return x*np.pi/180

    def cross(a, b, axis=0):
        valuesA = np.array([a[0].v, a[1].v, a[2].v])
        valuesB = np.array([b[0].v, b[1].v, b[2].v])
        uncA = np.array([a[0].u, a[1].u, a[2].u])    
        uncB = np.array([b[0].u, b[1].u, b[2].u])
        crossVal = np.cross(valuesA, valuesB, axis=0)
        crossUnc = np.cross(uncA, uncB, axis=0)

        result = np.ndarray(shape=(3,), dtype=np.object)
        for i in range(3):
            result[i] = Measurement(crossVal[i],crossUnc[i])
        return result

    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def nansum(array):
        try:
            unc = 0
            for i in np.nditer(array.u):
                if np.isfinite(i):
                    unc += i**2
            return Measurement(np.nansum(array.v), np.sqrt(unc))
        except AttributeError:
            return np.nansum(array)

    def nanmean(array):
        try:
            i = 0
            unc = 0
            val = np.nanmean(array.v)
            for u in np.nditer(array.u):
                if np.isfinite(u):
                    unc += u**2
                    i +=1
            return Measurement(val, np.sqrt(unc)/i)
        except AttributeError:
            return np.nanmean(array)
    def isfinite(array):
        try:
            return np.isfinite(array.v)
        except AttributeError:
            return np.isfinite(array)

    def nanmax(array):
        try:
            return np.nanmax(array.v)
        except AttributeError:
            return np.nanmax(array)

    def meshgrid(xRow, yRow):
        xg = Measurement(0,0)
        yg = Measurement(0,0)
        xg.v, yg.v = np.meshgrid(xRow.v, yRow.v, indexing='xy')
        xg.u, yg.u = np.meshgrid(xRow.u, yRow.u, indexing='xy')
        return xg, yg