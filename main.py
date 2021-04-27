import sys
import sqlite3
import pandas as pd
import numpy as np


# get data from sqlite
def get_data():
    try:
        conn = sqlite3.connect("database.sqlite")
    except sqlite3.Error as e:
        print(e)
    return conn


if __name__ == '__main__':
    # read data from squlite
    conn = get_data()
