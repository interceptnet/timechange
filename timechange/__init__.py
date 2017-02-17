#!/usr/bin/env python3
from collections import defaultdict
import sys
import numpy as np
import pandas
from PIL import Image

class TimeChange:
    def __init__(self, model=None):
        """Constructor"""
        self.training_files = defaultdict(set)
        self.model = model
    def add_training_file(self, label, filename):
        """Adds a training file to the dataset under a specific label
        Keyword arguments:
        label -- the label to store the filename under
        filename -- the filename to add to the database
        """
        #TODO: check if file exists
        self.training_files[label].add(filename)
    def remove_training_file(self, label, filename):
        """Removes a training file from a label
        Keyword arguments:
        label -- the label to store the filename under
        filename -- the filename to add to the database
        """
        try:
            self.training_files[label].remove(filename)
        except KeyError:
            print("File {} not found under label {}".format(filename, label), file=sys.stderr)
    def extract_features(self, time_series, method="fft", data_size=None):
        """Extracts features from a time series or array of time series and outputs an image
        Keyword arguments:
        time_series -- A numpy array or array of numpy arrays representing the time series data
        """
        #Fix data size
        if data_size is None:
            data_size = time_series.shape[1]
        #TODO: implement this
        if method == "fft":
            # Perform a fourier transform on the data
            features = np.abs(np.fft.rfft(time_series, n=data_size))
            #TODO: Extract complex features
            # Normalize the data against the maximum element
            return features / np.max(features)
        else:
            print("Feature extraction method {} invalid.".format(method), file=sys.stderr)
            return None
    def read_csv(self, filename, *args, **kwargs):
        """Reads a time-series data csv into a file
        Keyword arguments:
        filename -- filename to read from
        *args -- Extra args to pass the pandas parser
        """
        return pandas.read_csv(filename, *args, **kwargs).as_matrix().T
    def get_csv_columns(self, filename, *args, **kwargs):
        """Reads a csv file and returns the column names
        Keyword arguments:
        filename -- filename to read from
        *args -- Extra args to pass the pandas parser
        """
        return pandas.read_csv(filename, nrows=1, *args, **kwargs).columns
    def csv_to_image(self, filename, columns=None, method="fft", data_size=None):
        """Reads a csv file and returns the column names
        Keyword arguments:
        filename -- filename to read from
        columns -- columns to read. If this is set to None, use all
        method -- feature extraction method to use
        """
        # Set default columns if no argument specified
        if columns is None:
            columns = get_csv_columns(filename)
        # Read the csv into a numpy array
        data = self.read_csv(filename, usecols=columns)
        # Determine an FFT size
        if data_size is None:
            #Get the next nearest power of 2 to the data size
            data_size = 2 ** np.ceil(np.log2(data.shape[1]))
        #TODO: chunking
        # Extract features from the numpy array
        # Uses same variable name since data is not needed after feature extraction
        data = self.extract_features(data, method="fft", data_size=data_size)
        # Generate an image from the resulting feature representation
        # 
        return Image.fromarray(data * 255, 'L')
