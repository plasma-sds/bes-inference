# -*- coding: utf-8 -*-
"""
Unit tests for :class:`neuro_bes.data.besInferenceDatapoints`.

Run with:
    python -m unittest tests.test_data
    python -m unittest discover -s tests
"""

import unittest

import numpy as np

from neuro_bes.data import besInferenceDatapoints


class TestBesInferenceDatapoints(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        pass

    def tearDown(self):
        """Clean up after each test method."""
        pass

    # ------------------------------------------------------------------ #
    # __init__
    # ------------------------------------------------------------------ #
    def test_init(self):
        pass

    def test_init_from_path(self):
        pass

    # ------------------------------------------------------------------ #
    # add_datapoint
    # ------------------------------------------------------------------ #
    def test_add_datapoint(self):
        pass

    def test_add_datapoint_with_errors(self):
        pass

    def test_add_datapoint_resolution_mismatch(self):
        pass

    # ------------------------------------------------------------------ #
    # add_datapoints_bulk
    # ------------------------------------------------------------------ #
    def test_add_datapoints_bulk(self):
        pass

    def test_add_datapoints_bulk_with_errors(self):
        pass

    def test_add_datapoints_bulk_resolution_mismatch(self):
        pass

    def test_add_datapoints_bulk_count_mismatch(self):
        pass

    # ------------------------------------------------------------------ #
    # get_datapoints
    # ------------------------------------------------------------------ #
    def test_get_datapoints(self):
        pass

    def test_get_datapoints_include_errors(self):
        pass

    # ------------------------------------------------------------------ #
    # export_to_h5
    # ------------------------------------------------------------------ #
    def test_export_to_h5(self):
        pass


if __name__ == '__main__':
    unittest.main()
