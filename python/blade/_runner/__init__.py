from types import SimpleNamespace

import blade._internal as bl
from blade._runner_impl import runner as _runner

def runner(cls):
    class BasePipeline(cls):
        def __init__(self, *args, **kwargs):
            self.runner = _runner()

            self.input = SimpleNamespace()
            self.output = SimpleNamespace()
            self.module = SimpleNamespace()

            super().__init__(*args, **kwargs)

        def copy(self, dst, src):
            assert self.runner.copy(dst, src) == bl.result.success

        def enqueue(self, input_callback, result_callback, output_callback, input_id, output_id):
            res = self.runner.enqueue(input_callback, result_callback, output_callback, input_id, output_id)

            if res == bl.result.success or \
               res == bl.result.runner_queue_full or \
               res == bl.result.runner_queue_empty or \
               res == bl.result.runner_queue_none_available:
                return
            
            assert res == bl.result.success

        def dequeue(self, callback):
            res = self.runner.dequeue(callback)

            if res == bl.result.success or \
               res == bl.result.runner_queue_full or \
               res == bl.result.runner_queue_empty or \
               res == bl.result.runner_queue_none_available:
                return
            
            assert res == bl.result.success

        def __call__(self, *args):
            # Check if transfer_in and transfer_out functions exist.
            if not hasattr(self, 'transfer_in'):
                raise AttributeError("The 'transfer_in' function is missing in the pipeline.")
            if not hasattr(self, 'transfer_out'):
                raise AttributeError("The 'transfer_out' function is missing in the pipeline.")
            
            # Set all duets to the position #0.
            for namespace in [self.input, self.output]:
                for _, value in vars(namespace).items():
                    value.set(0)

            # Get the number of input elements of the transfer_in and transfer_out functions.
            num_args_transfer_in = self.transfer_in.__code__.co_argcount - 1
            num_args_transfer_out = self.transfer_out.__code__.co_argcount - 1

            # Check if the total number of arguments matches the expected count.
            total_expected_args = num_args_transfer_in + num_args_transfer_out
            if len(args) != total_expected_args:
                raise ValueError(f"Expected {total_expected_args} arguments, but got {len(args)}.")

            # Check if runner will output.
            will_output = self.runner.will_output()

            # Feed the first N elements to the transfer_in function.
            transfer_in_args = args[:num_args_transfer_in]
            self.transfer_in(*transfer_in_args)

            # Register the computation.
            assert self.runner.compute(0) == bl.result.success

            if will_output:
                # Feed the remaining elements to the transfer_out function.
                transfer_out_args = args[num_args_transfer_in:]
                self.transfer_out(*transfer_out_args)

            # Synchronize default stream.
            assert self.runner.synchronize(0) == bl.result.success

            return will_output
        
        def synchronize(self):
            assert self.runner.synchronize(0) == bl.result.success

    return BasePipeline
