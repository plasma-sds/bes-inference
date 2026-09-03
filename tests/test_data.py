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
        self.tags = ['Profile_' + str(i) for i in range(self.n_points)]

        # A ready-to-use, empty object built on the test grid.
        self.dataset = besInferenceDatapoints(
            grid=self.grid,
            energy=self.energy,
            species=self.species,
            ID=self.ID,
            zeff=self.zeff,
            q=self.q,
            temperature=self.temperature,
            verbose=self.verbose,
        )

    def tearDown(self):
        """Clean up after each test method."""
        self.dataset = None

    # ------------------------------------------------------------------ #
    # __init__
    # ------------------------------------------------------------------ #
    def test_init(self):
        # Grid and derived resolution.
        np.testing.assert_array_equal(self.dataset.grid, self.grid)
        self.assertEqual(self.dataset.resolution, self.resolution)

        # Scalar / string metadata is stored as provided.
        self.assertEqual(self.dataset.energy, self.energy)
        self.assertEqual(self.dataset.species, self.species)
        self.assertEqual(self.dataset.ID, self.ID)
        self.assertEqual(self.dataset.zeff, self.zeff)
        self.assertEqual(self.dataset.q, self.q)
        self.assertEqual(self.dataset.verbose, self.verbose)
        np.testing.assert_array_equal(self.dataset.temperature, self.temperature)

        # Datapoint containers start empty with the correct 2D shape.
        for arr in (self.dataset.densities, self.dataset.emissions,
                    self.dataset.density_errors, self.dataset.emission_errors):
            self.assertEqual(arr.shape, (0, self.resolution))
        self.assertEqual(self.dataset.tags, [])

    def test_init_defaults(self):
        # Optional numeric fields default to NaN when not provided.
        dataset = besInferenceDatapoints(grid=self.grid, ID=self.ID)
        self.assertTrue(np.isnan(dataset.energy))
        self.assertTrue(np.isnan(dataset.zeff))
        self.assertTrue(np.isnan(dataset.q))

        # Temperature defaults to an empty (0, resolution) block.
        self.assertEqual(dataset.temperature.shape, (0, self.resolution))

    def test_init_from_path(self):
        pass

    # ------------------------------------------------------------------ #
    # add_datapoint
    # ------------------------------------------------------------------ #
    def test_add_datapoint(self):
        density = self.densities[0]
        emission = self.emissions[0]
        tag = self.tags[0]

        self.dataset.add_datapoint(density, emission, tag)

        # One row was appended to each container.
        self.assertEqual(self.dataset.densities.shape, (1, self.resolution))
        self.assertEqual(self.dataset.emissions.shape, (1, self.resolution))
        self.assertEqual(self.dataset.tags, [tag])
        np.testing.assert_array_equal(self.dataset.densities[0], density)
        np.testing.assert_array_equal(self.dataset.emissions[0], emission)

        # Without explicit errors, all-NaN rows are stored.
        self.assertEqual(self.dataset.density_errors.shape, (1, self.resolution))
        self.assertEqual(self.dataset.emission_errors.shape, (1, self.resolution))
        self.assertTrue(np.all(np.isnan(self.dataset.density_errors[0])))
        self.assertTrue(np.all(np.isnan(self.dataset.emission_errors[0])))

    def test_add_datapoint_multiple(self):
        for i in range(self.n_points):
            self.dataset.add_datapoint(self.densities[i], self.emissions[i], self.tags[i])

        self.assertEqual(self.dataset.densities.shape, (self.n_points, self.resolution))
        self.assertEqual(self.dataset.emissions.shape, (self.n_points, self.resolution))
        self.assertEqual(self.dataset.tags, self.tags)
        np.testing.assert_array_equal(self.dataset.densities, self.densities)
        np.testing.assert_array_equal(self.dataset.emissions, self.emissions)

    def test_add_datapoint_with_errors(self):
        density = self.densities[0]
        emission = self.emissions[0]
        density_error = self.density_errors[0]
        emission_error = self.emission_errors[0]

        self.dataset.add_datapoint(density, emission, self.tags[0],
                                   density_error=density_error,
                                   emission_error=emission_error)

        self.assertEqual(self.dataset.density_errors.shape, (1, self.resolution))
        self.assertEqual(self.dataset.emission_errors.shape, (1, self.resolution))
        np.testing.assert_array_equal(self.dataset.density_errors[0], density_error)
        np.testing.assert_array_equal(self.dataset.emission_errors[0], emission_error)

    def test_add_datapoint_resolution_mismatch(self):
        bad_density = np.ones(self.resolution + 1)
        good_emission = self.emissions[0]

        with self.assertRaises(ValueError):
            self.dataset.add_datapoint(bad_density, good_emission, self.tags[0])

        # A mismatched emission also raises.
        with self.assertRaises(ValueError):
            self.dataset.add_datapoint(self.densities[0], np.ones(self.resolution - 1), self.tags[0])

    def test_add_datapoint_error_resolution_mismatch(self):
        # A provided error array with the wrong size raises.
        with self.assertRaises(ValueError):
            self.dataset.add_datapoint(self.densities[0], self.emissions[0], self.tags[0],
                                       density_error=np.ones(self.resolution + 1))

        with self.assertRaises(ValueError):
            self.dataset.add_datapoint(self.densities[0], self.emissions[0], self.tags[0],
                                       emission_error=np.ones(self.resolution + 1))

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
