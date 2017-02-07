"""Utilities for torchaudio."""
from __future__ import print_function

import fnmatch
import glob
import os


def get_files(directory, pattern, recursive=True):
    """ Return the full path to all files in directory matching the specified
    pattern.
    pattern should be a glob style pattern (e.g. "*.wav")
    """
    # This yields an iterator which really speeds up looking through
    # large, flat directories.
    if recursive is False:
        matches_iter = glob.iglob(os.path.join(directory, pattern))
        return matches_iter
    # If we want to recurse, use os.walk instead
    matches = list()
    for root, _, filenames in os.walk(directory):
        # pylint: disable=W0640
        matches.extend(map(lambda ss: os.path.join(root, ss),
                           fnmatch.filter(filenames, pattern)))
    return matches

#
# def _update_progress(progress):
#     print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(progress * 50),
#                                                   progress * 100), end="")
#
#
# def create_manifest(data_path, manifest_path, ordered=True):
#     file_paths = []
#     wav_files = [os.path.join(dirpath, f)
#                  for dirpath, dirnames, files in os.walk(data_path)
#                  for f in fnmatch.filter(files, '*.wav')]
#     size = len(wav_files)
#     counter = 0
#     for file_path in wav_files:
#         file_paths.append(file_path.strip())
#         counter += 1
#         _update_progress(counter / float(size))
#     print('\n')
#     if ordered:
#         _order_files(file_paths)
#     counter = 0
#     with io.FileIO(manifest_path, "w") as file:
#         for wav_path in file_paths:
#             transcript_path = wav_path.replace(
#                 '/wav/', '/txt/').replace('.wav', '.txt')
#             sample = os.path.abspath(wav_path) + ',' + \
#                 os.path.abspath(transcript_path) + '\n'
#             file.write(sample)
#             counter += 1
#             _update_progress(counter / float(size))
#     print('\n')
#
#
# def _order_files(file_paths):
#     print("Sorting files by length...")
#
#     def func(element):
#         output = subprocess.check_output(
#             ['soxi -D %s' % element.strip()],
#             shell=True
#         )
#         return float(output)
#
#     file_paths.sort(key=func)
