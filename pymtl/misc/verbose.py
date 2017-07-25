# /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = "Karl-Heinz Fiebig"
__copyright__ = "Copyright 2017"
__version__ = "$Revision"


OFF = 0
WARN = 1
ON = 2

print_level = WARN


def pyout(msg, lvl=ON):
    if lvl > print_level:
        print(msg)
