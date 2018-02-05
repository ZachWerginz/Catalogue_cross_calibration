import numpy as np
import uncertainty.measurement as unp

mtest1 = unp.Measurement((2,3), (1, 1))
ntest1 = np.array((5, 5))



def add_test():
    m1 = unp.Measurement(6, 2.3)
    m2 = unp.Measurement(3, 1.1)
    m_arr1 = unp.Measurement((6, 12), (2.3, 4.6))
    m_arr2 = unp.Measurement((3, 6), (1.1, 1.2))
    assert 2 + m1 == m1 + 2 and 2 + m1 == unp.Measurement(8, 2.3)
    assert m1 + m2 == m1 + m2 and m1 + m2 == unp.Measurement(6+3, np.sqrt(2.3**2 + 1.1**2))
    assert (m_arr1 + m_arr2 == m_arr1 + m_arr2).all() and \
           ((m_arr1 + m_arr2) == unp.Measurement((9, 18), (np.sqrt(2.3**2 + 1.1**2), np.sqrt(4.6**2 + 1.2**2)))).all()
    print("Add test succeeded")


def sub_test():
    m1 = unp.Measurement(6, 2.3)
    m2 = unp.Measurement(3, 1.1)
    m_arr1 = unp.Measurement((6, 12), (2.3, 4.6))
    m_arr2 = unp.Measurement((3, 6), (1.1, 1.2))
    assert m1 - 2 == unp.Measurement(3, 2.3)
    assert 2 - m1 == unp.Measurement(-3, 2.3)
    assert m1 - m2 == -(m2 - m1) and m1 - m2 == unp.Measurement(6-3, np.sqrt(2.3**2 + 1.1**2))
    assert m_arr1 - m_arr2 == -(m_arr1 - m_arr2).all() and \
        ((m_arr1 + m_arr2) == unp.Measurement((9, 18), (np.sqrt(2.3**2 + 1.1**2), np.sqrt(4.6**2 + 1.2**2)))).all()
    print("Add test succeeded")


def mult_test():
    m1 = unp.Measurement(6, 2.3)
    m2 = unp.Measurement(3, 1.1)
    m_arr1 = unp.Measurement((6, 12), (2.3, 4.6))
    m_arr2 = unp.Measurement((3, 6), (1.1, 1.2))
    assert m1*2 == 2*m1 and m1*2 == unp.Measurement(12, 4.6)
    assert m1*-2 == -2*m1 and m1*-2 == unp.Measurement(-12, 4.6)
    assert m1*m2 == m2*m1 and m1*m2 == unp.Measurement(18, np.sqrt((2.3/6)**2 + (1.1/3)**2))
    assert (m_arr1*2 == 2*m_arr1).all() and ((m_arr1 * 2) == unp.Measurement((12, 24), (4.6, 9.2))).all()
    print("Multiplication test succeeded")


def div_test():
    pass


def main():
    add_test()
    mult_test()

if __name__ == '__main__':
    main()
