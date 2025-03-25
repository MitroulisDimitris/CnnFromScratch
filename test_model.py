import unittest
import numpy as np
from train import Model  # Adjust import if the Model class is defined elsewhere

class TestModel(unittest.TestCase):
    def setUp(self):
        # Create a Model instance with a specified learning rate.
        self.model = Model(lr=0.01)
        self.batch_size = 4
        # Generate random input data with shape (batch_size, 1, 28, 28)
        self.x = np.random.randn(self.batch_size, 1, 28, 28)
        # Create dummy one-hot encoded labels (batch_size, 10)
        labels = np.random.randint(0, 10, size=self.batch_size)
        self.y = np.zeros((self.batch_size, 10))
        self.y[np.arange(self.batch_size), labels] = 1

    def test_forward_output_shape(self):
        """Test that forward pass returns an output with the correct shape."""
        output = self.model.forward(self.x, training=False)
        self.assertEqual(output.shape, (self.batch_size, 10),
                         "Output shape from forward pass should be (batch_size, 10).")

    def test_loss_numeric(self):
        """Test that compute_loss returns a non-negative float."""
        preds = self.model.forward(self.x, training=False)
        loss = self.model.compute_loss(preds, self.y)
        self.assertIsInstance(loss, float, "Loss should be a float value.")
        self.assertGreaterEqual(loss, 0, "Loss should be non-negative.")

    def test_weight_update(self):
        """
        Test that the parameters of layers that support weight updates
        are indeed changed after a training step.
        """
        # Capture initial weights for layers that have update_weights and a 'weights' attribute.
        weight_layers = []
        initial_weights = {}
        for idx, layer in enumerate(self.model.layers):
            if hasattr(layer, 'update_weights') and hasattr(layer, 'weights'):
                weight_layers.append((idx, layer))
                # Use .copy() to ensure we are not just storing a reference.
                initial_weights[idx] = layer.weights.copy()

        # Perform one training step.
        loss = self.model.train_on_batch(self.x, self.y)
        self.assertIsInstance(loss, float, "Training loss should be a float value.")
        # Check that the weights for each layer have been updated.
        for idx, layer in weight_layers:
            updated_weights = layer.weights
            # We expect the updated weights to be different from the initial weights.
            self.assertFalse(np.array_equal(initial_weights[idx], updated_weights),
                             f"Layer at index {idx} weights did not update.")

    def test_accuracy_evaluation(self):
        """Test that evaluate_accuracy returns a valid accuracy score between 0 and 1."""
        acc = self.model.evaluate_accuracy(self.x, self.y)
        self.assertIsInstance(acc, float, "Accuracy should be a float value.")
        self.assertGreaterEqual(acc, 0.0, "Accuracy should be at least 0.0.")
        self.assertLessEqual(acc, 1.0, "Accuracy should be at most 1.0.")

    def test_predict_probabilities_sum_to_one(self):
        """Test that the output of predict (probabilities) sums to 1 for each sample."""
        probs = self.model.predict(self.x)
        sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(sums, np.ones(self.batch_size), atol=1e-5,
                                   err_msg="Predicted probabilities for each sample should sum to 1.")

    def test_forward_consistency_in_inference(self):
        """
        Test that multiple forward passes in inference mode (where dropout is disabled)
        produce the same results.
        """
        output1 = self.model.forward(self.x, training=False)
        output2 = self.model.forward(self.x, training=False)
        np.testing.assert_allclose(output1, output2, atol=1e-5,
                                   err_msg="Forward pass in inference mode should be consistent.")

if __name__ == '__main__':
    unittest.main()
