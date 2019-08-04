from scipy.stats import truncnorm


# Use precomputation otherwise it is really slow
class TruncatedNormal:
    def __init__(self):
        self.dist = truncnorm(-4.0, 4.0, loc=0, scale=1.0)

        self.data = None
        self.count = 0

        self.precompute()

    def precompute(self):
        self.data = self.dist.rvs(10000)
        self.count = 0

    def __call__(self, mean, sd):
        if self.count == len(self.data):
            self.precompute()

        x = mean + sd * self.data[self.count]
        self.count += 1
        return x

