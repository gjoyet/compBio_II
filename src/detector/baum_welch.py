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

    def run(self, seqs: list[str]):
        pass

    def forward(self, seq: str) -> (ndarray, float):
        n = len(seq)
        f = np.zeros((10, n+1))
        f[to_index['S'], 0] = 1.0
        for i, char in enumerate(seq):
            for state in range(10):
                f[state, i+1] = self.ep[state, to_index[char]] * np.sum(self.tp[:, state] * f[:, i])
        p = np.sum(self.tp[:, to_index['E']] * f[:, n])
        return f, p

    def backward(self, seq: str) -> ndarray:
        n = len(seq)
        b = np.zeros((10, n+1))
        b[:, n] = self.tp[:, to_index['E']]
        for i, char in enumerate(reversed(seq)):
            for state in range(10):
                b[state, n-i-1] = np.sum(b[:, n-i] * self.ep[:, to_index[char]] * self.tp[state, :])
        return b

    def update(self):
        pass


if __name__ == '__main__':
    # TODO: remove seed when testing is over
    np.random.seed(1)
    seq = 'GGAACCAAGAACGAGGGGCAAGTGGGAGGAGGTGGTCACCTGGAGGGTGT' \
          'GGACCAGTGGTACACAGGTTAGGAGAGGGGGAAGGGCAGAGTTTACATTG'
    bw = BaumWelch(seq)
    f, p = bw.forward(seq)
    b = bw.backward(seq)
    pass
