"""
conftest.py

Shared pytest fixtures. `FakeLandmark` stands in for MediaPipe's
NormalizedLandmark -- it just needs .x/.y/.z/.visibility attributes, so
the pure-Python validation/geometry code can be tested without a real
model file or camera.
"""

from dataclasses import dataclass

import pytest


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


def make_landmarks(count: int = 33, visibility: float = 1.0):
    """A list of `count` landmarks, all at a distinct, harmless position."""
    return [FakeLandmark(x=0.5, y=0.5, visibility=visibility) for _ in range(count)]


@pytest.fixture
def fake_landmark():
    return FakeLandmark
