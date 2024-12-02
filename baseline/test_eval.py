import unittest
from config_utils import load_config, validate_config
from model_utils import load_model_and_tokenizer, generate_response
import os


class TestConfigUtils(unittest.TestCase):

    def setUp(self):
        self.valid_config = {
            "bbh_dir": "./data/bbh/",
            "flan_file": "./data/flan/flan.tsv",
            "BBH_MAX_TOKEN_LENGTH": 500,
            "FLAN_MAX_NEW_TOKENS": 50,
            "MAX_EXAMPLES": 10,
            "model_name": "meta-llama/Meta-Llama-3-8B"
        }
        self.invalid_config_missing_key = {
            "bbh_dir": "./data/bbh/",
            "BBH_MAX_TOKEN_LENGTH": 500
        }
        self.invalid_config_wrong_value = {
            "bbh_dir": "./data/bbh/",
            "flan_file": "./data/flan/flan.tsv",
            "BBH_MAX_TOKEN_LENGTH": -500,
            "FLAN_MAX_NEW_TOKENS": 50,
            "MAX_EXAMPLES": 10,
            "model_name": "meta-llama/Meta-Llama-3-8B"
        }

    def test_load_config_valid(self):
        config = load_config("./test_config.json")
        self.assertIsInstance(config, dict)

    def test_validate_config_valid(self):
        try:
            validate_config(self.valid_config)
        except ValueError:
            self.fail("validate_config() raised ValueError unexpectedly!")

    def test_validate_config_missing_key(self):
        with self.assertRaises(ValueError):
            validate_config(self.invalid_config_missing_key)

    def test_validate_config_wrong_value(self):
        with self.assertRaises(ValueError):
            validate_config(self.invalid_config_wrong_value)


class TestModelUtils(unittest.TestCase):

    def setUp(self):
        self.model_name = "gpt2"
        self.tokenizer, self.model = load_model_and_tokenizer(self.model_name)

    def test_load_model_and_tokenizer(self):
        self.assertIsNotNone(self.tokenizer)
        self.assertIsNotNone(self.model)

    def test_generate_response(self):
        prompt = "What is the capital of France?"
        response = generate_response(prompt, self.tokenizer, self.model, max_length=50)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


if __name__ == "__main__":
    unittest.main()
