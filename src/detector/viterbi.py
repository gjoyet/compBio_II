import numpy as np
from numpy import ndarray

to_index = {'A+': 0, 'C+': 1, 'G+': 2, 'T+': 3,
            'A-': 4, 'C-': 5, 'G-': 6, 'T-': 7}
to_state = {0: 'A+', 1: 'C+', 2: 'G+', 3: 'T+',
            4: 'A-', 5: 'C-', 6: 'G-', 7: 'T-'}
# Likelihood of transition from (row) to (column).
# Order: A+, C+, G+, T+, A-, C-, G-, T-.
trans_cpgi = np.array([[0.20871, 0.24734, 0.43353, 0.12142],
                       [0.15532, 0.35906, 0.29398, 0.19163],
                       [0.16185, 0.34990, 0.36959, 0.11866],
                       [0.10273, 0.34874, 0.34338, 0.20515]])
trans = np.concatenate((0.2*trans_cpgi, trans_cpgi))
trans = np.concatenate((trans, 1.5*trans), axis=1)

# Emission probabilities.
# Order: A+, C+, G+, T+, A-, C-, G-, T-.
emission = np.array(trans[0, :])


def viterbi(sequence: str) -> ndarray:
    # This is wrong: I need the transition / emission matrices AND
    # need to keep at least two columns of v at a time.

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
