import numpy as np
from numpy import ndarray


to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


class BaumWelch:

    def __init__(self, seqs: list[str], tol=1e-60):
        self.tp = np.array([[0.85, 0.15, 0], [0.15, 0.85, 0], [0.5, 0.5, 0]])
        self.n_states = self.tp.shape[0]
        p = np.random.normal(0.25, 0.01, (2, 4))
        self.ep = np.concatenate((p / np.sum(p, axis=1)[:, np.newaxis], np.zeros((1, 4))))
        self.seqs = seqs
        self.tol = tol
        pass

    def run(self, max_iter=300) -> int:
        counts = 0
        p = 1.0
        p_next = 0.0
        while abs(p_next - p) > self.tol and counts < max_iter:
            counts += 1
            p = p_next
            p_next = self.iterate()
            if counts % 10 == 0:
                print(p_next)
        return counts

    def iterate(self) -> float:
        A = np.zeros_like(self.tp)
        E = np.zeros_like(self.ep)
        ptot = 0

        for seq in seqs:
            f, pf = self.forward(seq)
            b, pb = self.backward(seq)
            assert np.isclose(pf, pb, rtol=1e-10, atol=0)
            ptot += pf

            for k in range(self.n_states):
                for l in range(self.n_states):
                    for i, char in enumerate(seq):
                        A[k, l] += f[k, i] * self.tp[k, l] * self.ep[l, to_index[char]] * b[l, i+1] / pf

            for k in range(self.n_states):
                for i, char in enumerate(seq):
                    E[k, to_index[char]] += f[k, i+1] * b[k, i+1] / pf

        ntp = np.sum(A, axis=1)[:, np.newaxis]
        nep = np.sum(E, axis=1)[:, np.newaxis]
        self.tp = np.divide(A, ntp, out=self.tp, where=ntp != 0)
        self.ep = np.divide(E, nep, out=self.ep, where=nep != 0)

        return ptot

    def forward(self, seq: str) -> (ndarray, float):
        L = len(seq)
        f = np.zeros((self.n_states, L + 1))
        f[-1, 0] = 1.0
        for i, char in enumerate(seq):
            for k in range(self.n_states-1):
                f[k, i+1] = self.ep[k, to_index[char]] * np.inner(f[:, i], self.tp[:, k])

        p = np.sum(f[:, -1])
        return f, p

    def backward(self, seq: str) -> (ndarray, float):
        L = len(seq)
        b = np.zeros((self.n_states, L + 1))
        b[:-1, -1] = 1.0
        for i, char in zip(range(L-1, 0, -1), reversed(seq[1:])):
            for k in range(self.n_states-1):
                b[k, i] = np.sum(b[:, i+1] * self.ep[:, to_index[char]] * self.tp[k, :])

        b[-1, 0] = np.sum(b[:, 1] * self.ep[:, to_index[seq[0]]] * self.tp[-1, :])
        p = b[-1, 0]
        return b, p


def read_seqs(filename, condition_nbr=1):
    seqs = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if 'Condition_{}\n'.format(condition_nbr) in l:
                seqs.append(lines[i+1].replace('\n', ''))

    return seqs


if __name__ == '__main__':
    seqs = read_seqs('../../SequencesCpG.txt', 1)
    bw = BaumWelch(seqs)
    its = bw.run()
    np.set_printoptions(precision=3)
    print('\nIterations until convergence with tolerance {}: {}\n'.format(bw.tol, its))
    print('Transition matrix:')
    print(bw.tp)
    print('')
    print('Emission probabilities:')
    print(bw.ep)
    print('')
