"""Personalized project uncertainty module.

This module attaches an uncertainty to variables by creating a new object Measurement that has a value and an
uncertainty. When performing operations the uncertainty should propogate as expected. This module had inspirations from
the uncertainties python package:

Uncertainties: a Python package for calculations with uncertainties, Eric O. LEBIGOT,
http://pythonhosted.org/uncertainties/

That package was not chosen because of performance issues since it does all the correlation calculations using
auto-differentiation and I do not.

"""

import numpy as np
np.seterr(invalid='ignore')


def cos(x):
    """Return the cosine of the input.

    Args:
        x: value or array of values

    Returns:
        result: cos(x)

    """
    try:
        return Measurement(np.cos(x.v), np.abs(np.sin(x.v) * x.u))
    except AttributeError:
        return np.cos(x)


def sin(x):
    """Return the sine of the input.

    Args:
        x: value or array of values

    Returns:
        result: sin(x)

    """
    try:
        return Measurement(np.sin(x.v), np.abs(np.cos(x.v) * x.u))
    except AttributeError:
        return np.sin(x)


def arctan(x):
    """Return the arctangent of the input.

    Args:
        x: value or array of values

    Returns:
        result: arctan(x)

    """
    try:
        val = np.arctan(x.v)
        unc = 1 / (1 + x.v * x.v) * x.u
        return Measurement(val, unc)
    except AttributeError:
        return np.arctan(x)


def arctan2(x, y):
    """Return the arctan2 of the input.

    Args:
        x: value or array of values

    Returns:
        result: arctan2(x)

    """
    try:
        val = np.arctan2(x.v, y.v)
        z = x / y
        unc = 1 / (1 + z.v * z.v) * z.u
        return Measurement(val, np.abs(unc))
    except AttributeError:
        return np.arctan2(x, y)


def arcsin(x):
    """Return the arcsin of the input.

    Args:
        x: value or array of values

    Returns:
        result: arcsin(x)

    """
    try:
        val = np.arcsin(x.v)
        unc = (1 / np.sqrt(1 - x.v * x.v)) * x.u
        return Measurement(val, np.abs(unc))
    except AttributeError:
        return np.arcsin(x)


def sqrt(x):
    """Return the square root of the input.

    Args:
        x: value or array of values

    Returns:
        result: sqrt(x)

    """
    try:
        return Measurement(np.sqrt(x.v), 0.5 * np.abs(x.u / np.sqrt(x.v)))
    except AttributeError:
        return np.sqrt(x)


def deg2rad(x):
    """Convert degrees to radians.

    Args:
        x: value or array of values

    Returns:
        result: x * pi/180

    """
    return x * np.pi / 180


def cross(a, b, axis=0):
    """Return the cross product of two vectors.

    Args:
        a: first vector
        b: second vector
        axis: which axis to take the cross product on

    Returns:
        result: cross(a, b)

    """
    values_a = np.array([a[0].v, a[1].v, a[2].v])
    values_b = np.array([b[0].v, b[1].v, b[2].v])
    unc_a = np.array([a[0].u, a[1].u, a[2].u])
    unc_b = np.array([b[0].u, b[1].u, b[2].u])
    cross_val = np.cross(values_a, values_b, axis=0)
    cross_unc = np.cross(unc_a, unc_b, axis=0)

    result = np.ndarray(shape=(3,), dtype=np.object)
    for i in range(3):
        result[i] = Measurement(cross_val[i], cross_unc[i])
    return result


def dot(a, b):
    """Return the dot product of two vectors element-wise.

    Args:
        a: first vector
        b: second vector

    Returns:
        result: dot(a, b) element-wise

    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def nansum(array):
    """Return the sum of an array ignoring nans.

    Args:
        array: array of values

    Returns:
        result: np.nansum(array)

    """
    try:
        unc = 0
        for i in np.nditer(array.u):
            if np.isfinite(i):
                unc += i ** 2
        return Measurement(np.nansum(array.v), np.sqrt(unc))
    except AttributeError:
        return np.nansum(array)


def nanmean(array):
    """Return the mean of an array ignoring nans.

    Args:
        array: array of values

    Returns:
        result: np.nanmean(array)

    """
    try:
        i = 0
        unc = 0
        if np.isnan(array.v).all() or len(array.v) == 0:
            return Measurement(np.nan, np.nan)
        val = np.nanmean(array.v)
        for u in np.nditer(array.u):
            if np.isfinite(u):
                unc += u ** 2
                i += 1
        return Measurement(val, np.sqrt(unc) / i)
    except AttributeError:
        if np.isnan(array).all() or len(array) == 0:
            return np.nan
        return np.nanmean(array)


def mean(array):
    """Return the mean of an array.

    Args:
        array: array of values

    Returns:
        result: np.mean(array)

    """
    try:
        i = 0
        unc = 0
        if np.isnan(array.v).all() or len(array.v) == 0:
            return Measurement(np.nan, np.nan)
        val = np.mean(array.v)
        for u in np.nditer(array.u):
            if np.isfinite(u):
                unc += u ** 2
                i += 1
        return Measurement(val, np.sqrt(unc) / i)
    except AttributeError:
        if np.isnan(array).all() or len(array) == 0:
            return np.nan
        return np.mean(array)


def isfinite(array):
    """Test for finiteness (not nan or infinite).

    Args:
        array: array of values

    Returns:
        result: np.isfinite(array)

    """
    try:
        return np.isfinite(array.v)
    except AttributeError:
        return np.isfinite(array)


def isnan(array):
    """Test for nan value.

    Args:
        array: array of values

    Returns:
        result: np.isnan(array)

    """
    try:
        return np.isnan(array.v)
    except AttributeError:
        return np.isnan(array)


def nanmax(array):
    """Find the max of the array ignoring nans.

    Args:
        array: array of values

    Returns:
        result: np.nanmax(array)

    """
    try:
        return np.nanmax(array.v)
    except AttributeError:
        return np.nanmax(array)


def nanmin(array):
    """Find the min of the array ignoring nans.

    Args:
        array: array of values

    Returns:
        result: np.nanmin(array)

    """
    try:
        return np.nanmin(array.v)
    except AttributeError:
        return np.nanmin(array)


def meshgrid(x_row, y_row):
    """Calculate a grid of values given an array of x-coordinates of a 2d array and the y-coordinates.

    Args:
        x_row: values for x-coordinates
        y_row: values for y-coordinates

    Returns:
        result: np.meshgrid(x_row, y_row)

    """
    xg = Measurement(0, 0)
    yg = Measurement(0, 0)
    xg.v, yg.v = np.meshgrid(x_row.v, y_row.v, indexing='xy')
    xg.u, yg.u = np.meshgrid(x_row.u, y_row.u, indexing='xy')
    return xg, yg


class Measurement:
    """This class is an extension of numpy's ndarray including uncertainty.

    Without subclassing ndarray, this adds the functionality of uncertainty to numpy's ndarray and all the required
    operations.

    Examples:
          >>> import uncertainty.measurement as mnp
          >>> a = mnp.Measurement(5, 3)
          >>> print(a)
          5 +/- 1

    """

    # Set array priority to override some of ndarray's ufunc binary relations
    __array_priority__ = 10000

    def __init__(self, value, uncertainty):
        self.v = np.asarray(value)
        self.u = np.asarray(uncertainty)

    def __getitem__(self, index):
        return Measurement(self.v[index], self.u[index])

    def __setitem__(self, index, item):
        self.v[index] = np.asarray(item)
        self.u[index] = np.asarray(item)

    def __repr__(self):
        return 'Measurement({}, {})'.format(self.v, self.u)

    def __str__(self):
        return '{} +/- {}'.format(self.v, self.u)

    def v(self):
        return self.v

    def u(self):
        return self.u

    def __len__(self):
        return len(self.v)

    def __add__(self, right):
        if isinstance(right, (float, np.ndarray, int)):
            return Measurement(self.v + right, self.u)
        unc = np.sqrt(self.u ** 2 + right.u ** 2)
        return Measurement(self.v + right.v, unc)

    def __radd__(self, left):
        return Measurement(self.v + left, self.u)

    def __iadd__(self, other):
        return self + other

    def __sub__(self, right):
        if isinstance(right, (float, np.ndarray, int)):
            return Measurement(self.v - right, self.u)
        unc = np.sqrt(self.u ** 2 + right.u ** 2)
        return Measurement(self.v - right.v, unc)

    def __rsub__(self, left):
        return Measurement(self.v*-1 + left, self.u*-1 + left)

    def __mul__(self, right):
        if isinstance(right, (float, np.ndarray, int)):
            val = self.v * right
            unc = np.abs(self.u * right)
            return Measurement(val, unc)
        val = self.v * right.v
        unc = np.sqrt((self.u / self.v) ** 2 + (right.u / right.v) ** 2)
        return Measurement(val, np.abs(unc * val))

    def __rmul__(self, left):
        return Measurement(left * self.v, np.abs(left * self.u))

    def __truediv__(self, right):
        if isinstance(right, (float, np.ndarray, int)):
            return Measurement(self.v / right, np.abs(self.u / right))
        unc = np.sqrt((self.u / self.v) ** 2 + (right.u / right.v) ** 2)
        val = self.v / right.v
        return Measurement(val, np.abs(unc * val))

    def __rtruediv__(self, left):
        return left * self ** (-1)

    def __pow__(self, power):
        val = self.v ** power
        unc = np.abs(val * power * self.u / self.v)
        return Measurement(val, unc)

    def __abs__(self):
        return Measurement(np.abs(self.v), self.u)

    def __eq__(self, a):
        return np.equal(self.v, a.v)

    def __gt__(self, b):
        try:
            return np.greater(self.v, b.v)
        except AttributeError:
            return np.greater(self.v, b)

    def __ge__(self, b):
        try:
            return np.greater_equal(self.v, b.v)
        except AttributeError:
            return np.greater_equal(self.v, b)

    def __lt__(self, b):
        try:
            return np.less(self.v, b.v)
        except AttributeError:
            return np.less(self.v, b)

    def __and__(self, b):
        try:
            return np.logical_and(self.v, b.v)
        except AttributeError:
            return np.logical_and(self.v, b)

    def __or__(self, b):
        try:
            return np.logical_or(self.v, b.v)
        except AttributeError:
            return np.logical_or(self.v, b)
