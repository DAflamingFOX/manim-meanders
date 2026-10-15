import random

import numpy as np
from manim import *  # pyright: ignore[reportWildcardImportFromLibrary]
from seaweed_algebra import Meander

from meander_element import MeanderElement


class MeanderWindDown(MovingCameraScene):
    def construct(self):

        m = Meander([17, 3], [10, 4, 6])
        # m = Meander([8, 5], [5, 6, 2])

        m_elem = MeanderElement(m)
        self.add(m_elem)

        m_elem.fit_camera(self)
        self.wait(1)

        m_elem.rotation_contraction_animation(self)

        # self.play(m_elem.rotation_contraction_animation(self))
