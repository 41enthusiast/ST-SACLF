import unittest


class MyTestCase(unittest.TestCase):
    def dataset_test(self):
        self.assertEqual(True, False)  # add assertion here

    def is_model_training(self):
        self.assertEqual(True, False)  # add assertion here

    def model_inference_test(self):
        self.assertEqual(True, False)  # add assertion here

    def model_metrics(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
