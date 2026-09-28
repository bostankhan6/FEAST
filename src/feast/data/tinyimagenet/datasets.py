"""TinyImageNet dataset wrapper.

Reuses the same ImageFolderTruncated implementation as CINIC-10 to avoid
duplication. Both datasets store images on disk in ImageFolder layout, so the
underlying class is identical.
"""

from feast.data.cinic10.datasets import ImageFolderTruncated  # noqa: F401
