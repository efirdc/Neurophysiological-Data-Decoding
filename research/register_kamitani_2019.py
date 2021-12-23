import copy

import nipy
from nipy.core.image.image import Image
from nipy.core.reference.coordinate_map import AffineTransform, CoordinateSystem

from .data.kamitani_2019 import Kamitani2019
from nipy.algorithms.registration import HistogramRegistration
from nipy.io.api import load_image


def main(root: str):
    dataset = Kamitani2019(root)
    subject = dataset.subjects[0]

    subject = dataset.subjects[0]
    session = subject['sessions']['natural_training'][0]
    t1 = subject['t1']
    t2 = session['t2']
    bold = session['runs'][0]

    t1_nipy = load_image(str(t1.path))
    t2_nipy = load_image(str(t2.path))
    bold_nipy = load_image(str(bold.path))

    t2_registration = HistogramRegistration(t2_nipy, bold_nipy)


if __name__ == '__main__':
    main(
        root="X:\\Datasets\\Deep-Image-Reconstruction\\"
    )


