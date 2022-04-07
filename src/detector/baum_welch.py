import numpy as np
from numpy import ndarray

to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
            'A+': 0, 'C+': 1, 'G+': 2, 'T+': 3,
            'A-': 4, 'C-': 5, 'G-': 6, 'T-': 7,
            'S': 8, 'E': 9}
to_state = {0: 'A+', 1: 'C+', 2: 'G+', 3: 'T+',
            4: 'A-', 5: 'C-', 6: 'G-', 7: 'T-'}


class BaumWelch:

    def __init__(self, seqs: list[str]):
        # transition probabilities from row to column (8x8 matrix)
        self.tp = np.random.randn(10, 10) * 0.01 + 1.0 / 8.0
        self.tp[:, to_index['S']] = np.zeros((10,))
        self.tp /= np.sum(self.tp, axis=1)[:, np.newaxis]
        self.tp[to_index['E'], :] = np.zeros((10,))
        # emission probabilities (8x4 matrix)
        self.ep = np.concatenate((np.eye(4), np.eye(4), np.zeros((2, 4))))
        # sequences
        self.seqs = seqs

    def run(self):
        for i in range(5):
            self.iterate()

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
        for k in range(10):
            for l in range(10):
                for j, s in enumerate(self.seqs):
                    for i, char in enumerate(s):
                        a[k, l] += (1.0 / all_p[j]) * all_f[j, k, i] * self.tp[k, l] * \
                                   self.ep[l, to_index[char]] * all_b[j, l, i+1]
        # TODO: fix issue in here (for now, commented it out since the emission probabilities are known in this problem)
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

    def forward(self, seq: str) -> (ndarray, float):
        n = len(seq)
        f = np.zeros((10, n + 1))
        f[to_index['S'], 0] = 1.0
        for i, char in enumerate(seq):
            for state in range(10):
                f[state, i + 1] = self.ep[state, to_index[char]] * np.sum(self.tp[:, state] * f[:, i])
        p = np.sum(self.tp[:, to_index['E']] * f[:, n])
        return f, p

    def backward(self, seq: str) -> ndarray:
        n = len(seq)
        b = np.zeros((10, n + 1))
        b[:, n] = self.tp[:, to_index['E']]
        for i, char in enumerate(reversed(seq[1:])):
            for state in range(10):
                b[state, n-i-1] = np.sum(b[:, n-i] * self.ep[:, to_index[char]] * self.tp[state, :])
        b[to_index['S'], 0] = np.sum(b[:, 1] * self.ep[:, to_index[seq[0]]] * self.tp[to_index['S'], :])
        return b


if __name__ == '__main__':
    # TODO: remove seed when testing is over
    np.random.seed(1)
    # seqs = ['GGAACCAAGAACGAGGGGCAAGTGGGAGGAGGTGGTCACCTGGAGGGTGTGGACCAGTGGTACACAGGTTAGGAGAGGGGGAAGGGCAGAGTTTACATTG',
    #        'CACTGGTGACTGTGGGCTGCATGGTGAAGGGGGGCAAAGCAGTGACAGCGGGAGTGAGCCCCTCTCAAAAACTGATGCCAACTACGCAGGACAGAGAGGG']
    seqs = ['ACGTACGT',
            'ACGTACGT']
    bw = BaumWelch(seqs)
    bw.run()
    print(bw.tp)
    print(bw.ep)
