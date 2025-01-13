import sys
import pdb
from accelerate import Accelerator

class ForkedPdb(pdb.Pdb):
    '''A Pdb subclass that may be used
    from a forked multiprocessing child
    '''

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def set_trace(accelerator: Accelerator):
    '''Set a tracepoint with a forked Pdb
    instance, so that it can be used in
    a multiprocessing child
    '''
    if accelerator.is_main_process:
        ForkedPdb().set_trace(sys._getframe().f_back)