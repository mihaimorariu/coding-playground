from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
import math

def sieve(up_to):
    cdef bint *primes = _sieve(up_to)
    response = [x for x in range(up_to + 1) if primes[x]]
    free(primes)
    return response

cdef bint *_sieve(int up_to):
    cdef int i, j
    cdef bint *primes = <bint *>malloc((up_to + 1) * sizeof(bint))

    for i in range(up_to + 1):
        primes[i] = 1

    primes[0] = primes[1] = False

    cdef int upper_limit = int(sqrt(up_to))
    for i in range(2, upper_limit + 1):
        if not primes[i]:
            continue
        j = 2 * i
        while j < up_to + 1:
            primes[j] = False
            j += i

    return primes
