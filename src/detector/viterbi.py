import numpy as np

to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
            'A+': 0, 'C+': 1, 'G+': 2, 'T+': 3,
            'A-': 4, 'C-': 5, 'G-': 6, 'T-': 7}
to_state = {0: 'A+', 1: 'C+', 2: 'G+', 3: 'T+',
            4: 'A-', 5: 'C-', 6: 'G-', 7: 'T-'}
# Likelihood of transition from (row) to (column).
# Order: A+, C+, G+, T+, A-, C-, G-, T-.
trans_cpgi = np.array()
trans_non_cpgi = np.array()

# Emission probabilities.
# Order: A+, C+, G+, T+, A-, C-, G-, T-.
emission_cpgi = np.array()
emission_non_cpgi = np.array()


def viterbi(sequence: str) -> list[str]:
    # This is wrong: I need the transition / emission matrices AND
    # need to keep at least two columns of v at a time.

    v_prev = 1.0
    ptr = ['S']

    for i in range(len(sequence)):
        v = np.zeros(8)
        char = sequence[i]
        cpgi = char + '+'
        no_cpgi = char + '-'
        v[to_index[cpgi]] = emission_cpgi[to_index[cpgi]] * v_prev * \
                                  trans_cpgi[to_index[ptr[-1]], to_index[char]]
        v[to_index[no_cpgi]] = emission_non_cpgi[to_index[no_cpgi]] * v_prev * \
                                  trans_non_cpgi[to_index[ptr[-1]], to_index[char]]
        v_prev = np.max(v)
        ptr.append(to_state[np.argmax(v)])

    ptr.append('E')
    return ptr
