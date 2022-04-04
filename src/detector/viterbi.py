import numpy as np

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


def viterbi(sequence: str) -> list[list[str]]:
    # This is wrong: I need the transition / emission matrices AND
    # need to keep at least two columns of v at a time.

    n = len(sequence)
    char = sequence[0]
    v = np.zeros((8, n))
    ptr = ['S']
    v[to_index[char + '+'], 0] = emission[to_index[char + '+']] / 8.0
    v[to_index[char + '-'], 0] = emission[to_index[char + '-']] / 8.0

    for i in range(1, n):
        char = sequence[i]
        v[to_index[char + '+'], i] = emission[to_index[char + '+']] * np.max(v[:, i-1] * trans[:, to_index[char + '+']])
        v[to_index[char + '-'], i] = emission[to_index[char + '-']] * np.max(v[:, i-1] * trans[:, to_index[char + '-']])
        max_state = np.argmax(v[:, i])
        ptr.append(to_state[np.argmax(v[:, i - 1] * trans[:, max_state])])

    ptr.append(to_state[np.argmax(v[:, -1])])
    ptr.append('E')
    return ptr


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
    print(ptr)
