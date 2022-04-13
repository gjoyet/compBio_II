import numpy as np
from numpy import ndarray


to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
to_state = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


class BaumWelch:

    def __init__(self, seqs: list[str], tol=10 ** -8):
        # transition probabilities from row to column
        self.tp = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.end_p = np.ones((2,))
        self.start_p = np.array([0.5, 0.5])
        # emission probabilities
        self.ep = np.array([[0.2, 0.3, 0.3, 0.2], [0.25, 0.25, 0.25, 0.25]])
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
        for k in range(2):
            for l in range(2):
                for j, s in enumerate(self.seqs):
                    for i, char in enumerate(s[:-1]):
                        a[k, l] += (1.0 / all_p[j]) * all_f[j, k, i] * self.tp[k, l] * \
                                   self.ep[l, to_index[char]] * all_b[j, l, i+1]
        e = np.zeros_like(self.ep)
        for k in range(2):
            for beta in range(4):
                for j, s in enumerate(self.seqs):
                    for i, char in enumerate(s[:-1]):
                        if to_index[char] == beta:
                            e[k, beta] += (1.0 / all_p[j]) * all_f[j, k, i+1] * all_b[j, k, i+1]
        s = np.zeros_like(self.start_p)
        for k in range(2):
            for j, s in enumerate(self.seqs):
                s = all_b[j, :, 0] * self.ep[:, to_index[s[0]]] * self.start_p

        ntp = np.sum(a, axis=1)[:, np.newaxis]
        nep = np.sum(e, axis=1)[:, np.newaxis]
        nsp = np.sum(s)
        self.tp = np.divide(a, ntp, out=self.tp, where=ntp != 0)
        self.ep = np.divide(e, nep, out=self.ep, where=nep != 0)
        self.start_p = np.divide(s, nsp, out=self.start_p, where=nsp != 0)
        return np.sum(all_p)

    def forward(self, seq: str) -> (ndarray, float):
        n = len(seq)
        f = np.zeros((2, n))
        f[:, 0] = self.start_p * self.ep[:, to_index[seq[0]]]
        for i, char in enumerate(seq[1:]):
            for state in range(2):
                f[state, i + 1] = self.ep[state, to_index[char]] * np.sum(self.tp[:, state] * f[:, i])
        p = np.sum(self.end_p * f[:, -1])
        return f, p

    def backward(self, seq: str) -> ndarray:
        n = len(seq)
        b = np.zeros((2, n))
        b[:, -1] = self.end_p
        for i, char in enumerate(reversed(seq[1:])):
            for state in range(2):
                b[state, -(i+2)] = np.sum(b[:, -(i+1)] * self.ep[:, to_index[char]] * self.tp[state, :])
        return b


if __name__ == '__main__':
    # TODO: remove seed when testing is over
    np.random.seed(1)
    seqs = ['GGAACCAAGAACGAGGGGCAAGTGGGAGGAGGTGGTCACCTGGAGGGTGTGGACCAGTGGTACACAGGTTAGGAGAGGGGGAAGGGCAGAGTTTACATTG',
            'CACTGGTGACTGTGGGCTGCATGGTGAAGGGGGGCAAAGCAGTGACAGCGGGAGTGAGCCCCTCTCAAAAACTGATGCCAACTACGCAGGACAGAGAGGG']
    # seqs = ['ACGTACGT',
    #         'CCAGTAGT']
    bw = BaumWelch(seqs)
    c = bw.run()
    np.set_printoptions(precision=3)
    print('\nIterations until convergence with tolerance {}: {}\n'.format(bw.tol, c))
    print('Transition matrix:')
    print(bw.tp)
    print('')
    print('Emission probabilities:')
    print(bw.ep)
    print('')
    print('Probabilities for initial state:')
    print(bw.start_p)
