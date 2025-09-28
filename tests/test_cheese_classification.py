import unittest
from scripts.train import train
from scripts.generate import generate

class TestCheeseClassification(unittest.TestCase):

    def test_train(self):
        """Test the train function with a mock configuration."""
        # Mock configuration
        cfg = {
            "model": {"instance": "mock_model"},
            "optim": {},
            "datamodule": {},
            "epochs": 1,
            "checkpoint_path": "mock_checkpoint.pt"
        }
        try:
            train(cfg)
        except Exception as e:
            self.fail(f"train() raised {e}")

    def test_generate(self):
        """Test the generate function with a mock configuration."""
        # Mock configuration
        cfg = {
            "dataset_generator": {"output_dir": "mock_output"},
            "labels_file": "mock_labels.txt"
        }
        try:
            generate(cfg)
        except Exception as e:
            self.fail(f"generate() raised {e}")

if __name__ == "__main__":
    unittest.main()