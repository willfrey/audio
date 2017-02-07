"""A number of useful transforms for audio data."""

# pylint: disable=R0903


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose

    Example:
        >>> transforms.Compose([])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data
