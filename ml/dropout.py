import numpy as np


class Dropout:
    def __init__(self, dropout_rate=0.5):
        assert 0.0 <= dropout_rate <= 1.0, "Dropout rate must be between 0 and 1."
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, inputs, training=True):
        if training:
            # Create a mask with 1s where neurons are active
            self.mask = (np.random.rand(*inputs.shape) > self.dropout_rate).astype(
                np.float32
            )
            print(self.mask)
            input()
            # Scale the output to maintain expected sum of activations
            return inputs * self.mask / (1.0 - self.dropout_rate)
        else:
            # During inference, don't apply dropout
            return inputs

    def backward(self, d_out):
        return d_out * self.mask / (1.0 - self.dropout_rate)


if __name__ == "__main__":
    np.random.seed(42)
    inputs = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
    )

    dropout = Dropout(dropout_rate=0.5)

    # Forward pass during training
    training_outputs = dropout.forward(inputs, training=True)
    print("Forward pass (training):")
    print(training_outputs)

    # Forward pass during inference
    inference_outputs = dropout.forward(inputs, training=False)
    print("\nForward pass (inference):")
    print(inference_outputs)

    # Backward pass
    d_out = np.ones_like(inputs)  # Example gradient (ones matrix)
    gradients = dropout.backward(d_out)
    print("\nBackward pass (gradients):")
    print(gradients)
