# Import the blade library
import blade as bl

# Define the asynchronous pipeline class
@bl.runner
class Pipeline:
    # Initialize the pipeline with shape and configuration
    def __init__(self, shape, config):
        # Create input and output buffers with the specified shape
        self.input.buf = bl.array_tensor(shape, dtype=bl.cf32)
        self.output.buf = bl.array_tensor(shape, dtype=bl.cf32)

        # Initialize the polarizer module with the given configuration and input buffer
        self.module.polarizer = bl.module(bl.polarizer, config, self.input.buf)

    # Transfer data from the provided buffer to the input buffer
    def transfer_in(self, buf):
        self.copy(self.input.buf, buf)

    # Transfer data from the polarizer's output to the output buffer
    def transfer_result(self):
        self.copy(self.output.buf, self.module.polarizer.get_output())

    # Transfer data from the output buffer to the provided buffer
    def transfer_out(self, buf):
        self.copy(buf, self.output.buf)

# Define the shape and configuration for the pipeline
shape = (2, 192, 8750, 2)
config = {
    'inputPolarization': bl.pol.xy,
    'outputPolarization': bl.pol.lr,
}

# Create an instance of the pipeline
pipeline = Pipeline(shape, config)

# Create host input and output buffers with the specified shape and device
host_input = bl.array_tensor(shape, dtype=bl.cf32, device=bl.cpu)
host_output = bl.array_tensor(shape, dtype=bl.cf32, device=bl.cpu)

# Initialize enqueue and dequeue counters
enqueue_count = [0]
dequeue_count = [0]
iterations = [0]

# Loop until 8 items have been dequeued
while iterations[0] < 8:
    # Define the input callback function
    def input_callback():
        enqueue_count[0] += 1
        pipeline.transfer_in(host_input)
        return bl.result.success

    # Define the result callback function
    def result_callback():
        pipeline.transfer_result()
        return bl.result.success

    # Define the output callback function
    def output_callback():
        dequeue_count[0] += 1
        pipeline.transfer_out(host_output)
        return bl.result.success

    # Enqueue the pipeline with the input and output callbacks and the current enqueue count
    pipeline.enqueue(input_callback, result_callback, output_callback, enqueue_count[0], dequeue_count[0])

    # Define the dequeue callback function
    def callback(input_id, output_id, did_output):
        print(f"[D] Input ID: {input_id}, Output ID: {output_id}, Did Output: {did_output}")
        if did_output:
            iterations[0] += 1
        return bl.result.success

    # Dequeue the pipeline with the callback function
    pipeline.dequeue(callback)
