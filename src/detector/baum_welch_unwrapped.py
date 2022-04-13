import numpy as np
from numpy import ndarray

to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
            'A+': 0, 'C+': 1, 'G+': 2, 'T+': 3,
            'A-': 4, 'C-': 5, 'G-': 6, 'T-': 7,
            'S': 8}
to_state = {0: 'A+', 1: 'C+', 2: 'G+', 3: 'T+',
            4: 'A-', 5: 'C-', 6: 'G-', 7: 'T-'}


class BaumWelchUnwrapped:

    def __init__(self, seqs: list[str], tol=10**-8):
        # transition probabilities from row to column
        self.tp = np.random.randn(9, 9) * 0.01 + 1.0 / 8.0
        self.tp[:, to_index['S']] = np.zeros((9,))
        self.tp /= np.sum(self.tp, axis=1)[:, np.newaxis]
        self.tp_to_end = np.ones((9, ))
        # emission probabilities
        self.ep = np.concatenate((np.eye(4), np.eye(4), np.zeros((1, 4))))
        # sequences
        self.seqs = seqs
        self.tol = tol

    def run(self):
        p = 1.0
        p_next = 0.0
        counts = 0
        while abs(p - p_next) > self.tol or counts <= 100:
            counts += 1
            p = p_next
            p_next = self.iterate()
        return counts

    def iterate(self):
        all_f = []
        all_p = []
        all_b = []
        for s in self.seqs:
            f, p = self.forward(s)
            all_f.append(f)
            all_p.append(p)
            all_b.append(self.backward(s))
        all_f = np.array(all_f)
        all_p = np.array(all_p)
        all_b = np.array(all_b)

        a = np.zeros_like(self.tp)
        for k in range(9):
            for l in range(9):
                for j, s in enumerate(self.seqs):
                    for i, char in enumerate(s):
                        a[k, l] += (1.0 / all_p[j]) * all_f[j, k, i] * self.tp[k, l] * \
                                   self.ep[l, to_index[char]] * all_b[j, l, i+1]
        e = np.zeros_like(self.ep)
        for k in range(8):
            for beta in range(4):
                for j, s in enumerate(self.seqs):
                    for i, char in enumerate(s):
                        if to_index[char] == beta:
                            e[k, beta] += (1.0 / all_p[j]) * all_f[j, k, i+1] * all_b[j, k, i+1]

        ntp = np.sum(a, axis=1)[:, np.newaxis]
        nep = np.sum(e, axis=1)[:, np.newaxis]
        self.tp = np.divide(a, ntp, out=self.tp, where=ntp != 0)
        self.ep = np.divide(e, nep, out=self.ep, where=nep != 0)
        return np.sum(all_p)

    def forward(self, seq: str) -> (ndarray, float):
        n = len(seq)
        f = np.zeros((9, n + 1))
        f[to_index['S'], 0] = 1.0
        for i, char in enumerate(seq):
            for state in range(9):
                f[state, i + 1] = self.ep[state, to_index[char]] * np.sum(self.tp[:, state] * f[:, i])
        p = np.sum(self.tp_to_end * f[:, n])
        return f, p

    def backward(self, seq: str) -> ndarray:
        n = len(seq)
        b = np.zeros((9, n + 1))
        b[:, n] = self.tp_to_end
        for i, char in enumerate(reversed(seq[1:])):
            for state in range(9):
                b[state, n-i-1] = np.sum(b[:, n-i] * self.ep[:, to_index[char]] * self.tp[state, :])
        b[to_index['S'], 0] = np.sum(b[:, 1] * self.ep[:, to_index[seq[0]]] * self.tp[to_index['S'], :])
        return b


if __name__ == '__main__':
    # TODO: remove seed when testing is over
    np.random.seed(1)
    seqs = ['GGAACCAAGAACGAGGGGCAAGTGGGAGGAGGTGGTCACCTGGAGGGTGTGGACCAGTGGTACACAGGTTAGGAGAGGGGGAAGGGCAGAGTTTACATTG',
            'CACTGGTGACTGTGGGCTGCATGGTGAAGGGGGGCAAAGCAGTGACAGCGGGAGTGAGCCCCTCTCAAAAACTGATGCCAACTACGCAGGACAGAGAGGG']
    # seqs = ['ACGTACGT',
    #         'CCAGTAGT']
    bw = BaumWelchUnwrapped(seqs)
    c = bw.run()
    np.set_printoptions(precision=3)
    print('\nIterations until convergence with tolerance {}: {}\n'.format(bw.tol, c))
    print('Transition matrix:')
    print(bw.tp)
    print('')
    print('Emission probabilities:')
    print(bw.ep)
