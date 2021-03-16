#!/usr/bin/env python3
"""Program that created a pd.DataFrame from a dictionary"""
import pandas as pd


dict = {'First': [i / 2 for i in range(0, 4)],
        'Second': ['one', 'two', 'three', 'four']}
df = pd.DataFrame(dict, list('ABCD'))
