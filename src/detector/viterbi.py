import numpy as np
from numpy import ndarray

to_index = {'A+': 0, 'C+': 1, 'G+': 2, 'T+': 3,
            'A-': 4, 'C-': 5, 'G-': 6, 'T-': 7}
to_state = {0: 'A+', 1: 'C+', 2: 'G+', 3: 'T+',
            4: 'A-', 5: 'C-', 6: 'G-', 7: 'T-'}
# Likelihood of transition from (row) to (column).
# Order: A+, C+, G+, T+, A-, C-, G-, T-.
trans = np.array()

# Emission probabilities.
# Order: A+, C+, G+, T+, A-, C-, G-, T-.
emission = np.array()


def viterbi(sequence: str) -> ndarray:
    n = len(sequence)
    char = sequence[0]
    v = np.zeros((8, n))
    ptr = np.zeros((8, n+1))
    ptr[:, 0] = -np.ones((8,))
    v[to_index[char + '+'], 0] = emission[to_index[char + '+']] / 8.0
    v[to_index[char + '-'], 0] = emission[to_index[char + '-']] / 8.0

    for i in range(1, n):
        char = sequence[i]
        v[to_index[char + '+'], i] = emission[to_index[char + '+']] * np.max(v[:, i-1] * trans[:, to_index[char + '+']])
        v[to_index[char + '-'], i] = emission[to_index[char + '-']] * np.max(v[:, i-1] * trans[:, to_index[char + '-']])
        for k in range(8):
            ptr[k, i] = np.argmax(v[:, i - 1] * trans[:, k])

    ptr[:, n] = np.argmax(v[:, -1])
    return ptr.astype(int)


def traceback(ptr: ndarray) -> list[str]:
    n = ptr.shape[1]
    path = ['E']
    opt = ptr[0, n-len(path)]

    while opt >= 0:
        path.insert(0, to_state[opt])
        opt = ptr[opt, n-len(path)]

    path.insert(0, 'S')
    return path


if __name__ == '__main__':
    seq = 'GGAACCAAGAACGAGGGGCAAGTGGGAGGAGGTGGTCACCTGGAGGG' \
          'TGTGGACCAGTGGTACACAGGTTAGGAGAGGGGGAAGGGCAGAGTTT' \
          'ACATTGCCCGTATGCTGGCGAGTGAAGTCCACTAGGAACTGAGACAT' \
          'GAACTTGAGGCTTAGCAAAAGAGAGCGACTTAGAGAAAGAGCACCCG' \
          'CACTGGTGACTGTGGGCTGCATGGTGAAGGGGGGCAAAGCAGTGACA' \
          'GCGGGAGTGAGCCCCTCTCAAAAACTGATGCCAACTACGCAGGACAG' \
          'AGAGGGGGCGGGGAAGGGGGAGTGACCTGAGGGAGACTGGGGCTCAA' \
          'GAAAAGCCTTTTTGTGTTGGTTGTTTTAAAGGCTGGCGATACTGTAG' \
          'CATGCTTAGTTCTAAGGAGAGGAA'
    ptr = viterbi(seq)
    path = traceback(ptr)
    print(path)
