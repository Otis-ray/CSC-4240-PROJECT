"""Regression coverage for interactive demo-recording selection."""
import unittest
from unittest.mock import patch

from Offline_app.__main__ import _select_wav


class WAVSelectionTests(unittest.TestCase):
    @patch('builtins.input', side_effect=['1'])
    def test_speaker_one(self, _):
        self.assertEqual(_select_wav(), 'Filters/dataset/noisy/Speaker1.wav')

    @patch('builtins.input', side_effect=['2'])
    def test_asaph_demo(self, _):
        self.assertEqual(_select_wav(), '48kBounce_filt.wav')

    @patch('builtins.input', side_effect=[''])
    def test_default_asaph_demo(self, _):
        self.assertEqual(_select_wav(), '48kBounce_filt.wav')

    @patch('builtins.input', side_effect=['3', '/tmp/mytest.wav'])
    def test_custom_absolute_path(self, _):
        self.assertEqual(_select_wav(), '/tmp/mytest.wav')

    @patch('builtins.input', side_effect=['5', '3', '', 'relative.wav'])
    def test_reprompt_invalid_choice_and_empty_path(self, _):
        self.assertEqual(_select_wav(), 'relative.wav')


if __name__ == '__main__':
    unittest.main()
