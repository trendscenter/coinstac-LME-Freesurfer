#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

def saveBin(path, arr):
    with open(path, 'wb+') as fh:
        header = '%s' % str(arr.dtype)
        for index in arr.shape:
            header += ' %d' % index
        header += '\n'
        fh.write(header.encode())
        fh.write(arr.data.tobytes())
        os.fsync(fh)


def loadBin(path):
    with open(path, 'rb') as fh:
        header = fh.readline().decode().split()
        dtype = header.pop(0)
        arrayDimensions = []
        for dimension in header:
            arrayDimensions.append(int(dimension))
        arrayDimensions = tuple(arrayDimensions)
        return np.frombuffer(fh.read(), dtype=dtype).reshape(arrayDimensions)

def list_recursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in list_recursive(v, key):
                yield found
        if k == key:
            yield v
