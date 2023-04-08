import faulthandler

from metaflow import FlowSpec, Parameter, step
from neighbors import run

faulthandler.enable()

class Pipeline(FlowSpec):
    """Simple Metaflow pipeline exercising the segfaulting `import torch` → `cuml.NearestNeighbors` code path.

    I've so far not reproduced the segfault outside of Metaflow; I think something about its step-cleanup process is
    either causing or leaving time for a memory error related to `cuml` data-structures.
    """
    random_seed = Parameter('random_seed', type=int, default=123)
    shape = Parameter('shape', type=str, default='10x2')

    @step
    def start(self):
        run(random_seed=self.random_seed, shape=self.shape)
        print('finished run, calling self.next')
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    Pipeline()
