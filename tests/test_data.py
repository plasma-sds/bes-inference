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
        # Spatial grid of 50 points along the beam in [m].
        self.resolution = 50
        self.n_points = 10
        self.grid = np.linspace(0.0, 1.0, self.resolution)

        # Metadata for the object.
        self.energy = 60
        self.species = 'Li'
        self.ID = 'test_dataset'
        self.zeff = 1.5
        self.q = 2.0
        self.verbose = 'Dataset used for unit testing.'
        self.temperature = np.full((self.n_points, self.resolution), 1.0e3)

        # 10 density-emission pairs (50 points each) with matching errors.
        rng = np.random.default_rng(seed=42)
        self.densities = rng.uniform(1.0e18, 1.0e19, size=(self.n_points, self.resolution))
        self.emissions = rng.uniform(0.0, 1.0, size=(self.n_points, self.resolution))
        self.density_errors = 0.1 * self.densities
        self.emission_errors = 0.1 * self.emissions
        self.tags = ['point_' + str(i) for i in range(self.n_points)]

    def tearDown(self):
        """Clean up after each test method."""
        self.dataset = None

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
