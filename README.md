torch-audio
============

This repository consists of:

- [torchaudio.data](#data) : Generic data loaders for audio
- [torchaudio.datasets](#datasets) : Pre-built loaders for common audio Datasets

Installation
============

From source:

.. code:: bash

    python setup.py install

Datasets
========

The following dataset loaders are available:

-  `AN4 Audio Database<#an4>`__
-  `LibriSpeech <#libri>`__

AN4
~~~~~
``dset.MNIST(root, train=True, transform=None, target_transform=None, download=False)``

``root``: root directory of dataset where ``processed/training.pt`` and ``training/test.pt`` exist

``train``: ``True`` - use training set, ``False`` - use test set.

``transform``: transform to apply to input images

``target_transform``: transform to apply to targets (class labels)

``download``: whether to download the MNIST data


LibriSpeech
~~~~

This requires the `COCO API to be
installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`__

Captions:
^^^^^^^^^

``dset.CocoCaptions(root="dir where images are", annFile="json annotation file", [transform, target_transform])``

Example:

.. code:: python

    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    cap = dset.CocoCaptions(root = 'dir where images are',
                            annFile = 'json annotation file',
                            transform=transforms.ToTensor())

    print('Number of samples: ', len(cap))
    img, target = cap[3] # load 4th sample

    print("Image Size: ", img.size())
    print(target)

Output:

::

    Number of samples: 82783
    Image Size: (3L, 427L, 640L)
    [u'A plane emitting smoke stream flying over a mountain.',
    u'A plane darts across a bright blue sky behind a mountain covered in snow',
    u'A plane leaves a contrail above the snowy mountain top.',
    u'A mountain that has a plane flying overheard in the distance.',
    u'A mountain view with a plume of smoke in the background']

Detection:
^^^^^^^^^^

``dset.CocoDetection(root="dir where images are", annFile="json annotation file", [transform, target_transform])``