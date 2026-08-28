"""Spatial region and lattice construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import DEFAULT_RADIUS


@dataclass(frozen=True)
class Sphere:
    """A spherical region of vacuum.

    Parameters
    ----------
    radius
        Sphere radius in metres. Defaults to 0.5 um, i.e. a diameter of 1 um.
    """

    radius: float = DEFAULT_RADIUS

    @property
    def volume(self) -> float:
        """Enclosed volume [m^3]."""
        return 4.0 / 3.0 * np.pi * self.radius**3

    @property
    def surface_area(self) -> float:
        """Bounding surface area [m^2]."""
        return 4.0 * np.pi * self.radius**2

    def lattice_spacing(self, n_per_radius: int) -> float:
        """Lattice spacing [m] for a given resolution."""
        return self.radius / n_per_radius

    def lattice_points(self, n_per_radius: int) -> np.ndarray:
        """Cubic lattice clipped to the sphere.

        Parameters
        ----------
        n_per_radius
            Number of lattice spacings per radius; the spacing is
            radius / n_per_radius.

        Returns
        -------
        ndarray of shape (n_sites, 3)
            Cartesian coordinates in metres.

        Notes
        -----
        Site count grows as n_per_radius**3 while dense diagonalisation of the
        resulting correlation matrix costs O(n_sites**3). Measured timings are
        recorded in docs/computational-limits.md; n_per_radius above about 16
        is not tractable on a workstation.
        """
        if n_per_radius < 1:
            raise ValueError("n_per_radius must be at least 1")
        a = self.lattice_spacing(n_per_radius)
        axis = np.arange(-n_per_radius, n_per_radius + 1) * a
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        points = grid.reshape(-1, 3)
        inside = np.sum(points**2, axis=1) <= self.radius**2 * (1 + 1e-12)
        return points[inside]


def pair_distances(points: np.ndarray) -> np.ndarray:
    """Full matrix of Euclidean separations between points [m]."""
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))
