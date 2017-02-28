#!/usr/bin/env python3
from collections import defaultdict
import sys
import os
import numpy as np
import pandas
from PIL import Image
from . import transform

class TimeChange:
    def __init__(self, project_name="default", parent_folder=None):
        """Constructor
        """
        #TODO:better default name to avoid collision 
        #Store the project's name
        self.project_name = project_name
        #Sets the project parent folder
        if parent_folder is None:
            parent_folder = os.path.expanduser("~/.timechange")
        #Stores where the project profile will be kept
        self.project_path = os.path.join(parent_folder, project_name)
        #Stores what csv columns to use
        self.columns = None #Default values
    def add_training_file(self, label, filename):
        """Adds a training file to the dataset under a specific label
        Keyword arguments:
        label -- the label to store the filename under
        filename -- the filename to add to the database
        """
        #TODO: implement this
        print("DUMMY: {} REMOVED FROM {}".format(filename, label))
    def remove_training_file(self, label, filename):
        """Removes a training file from a label
        Keyword arguments:
        label -- the label to store the filename under
        filename -- the filename to add to the project
        """
        #TODO: implement this
        print("DUMMY: {} REMOVED FROM {}".format(filename, label))
    def get_csv_columns(self, filename, *args, **kwargs):
        """Reads a csv file and returns the column names
        Keyword arguments:
        filename -- filename to read from
        *args -- Extra args to pass the pandas parser
        """
        return list(pandas.read_csv(filename, nrows=1, *args, **kwargs).columns)
    def convert_csv(self, input_filename, output_filename=None, method="fft", chunk_size=32):
        """Reads a csv file and returns the column names
        Keyword arguments:
        input_filename -- CSV filename to read from
        output_filename -- png file to output to. Uses a standard scheme if None
        columns -- columns to read. If this is set to None, use all
        method -- feature extraction method to use
        """
        # Set default columns if no argument specified
        if self.columns is None:
            self.columns = self.get_csv_columns(input_filename)
        # Set default filename if no argument specified
        if output_filename is None:
            input_path = os.path.split(input_filename)
            input_path[-1] = "converted_{}.png".format(input_path[-1])
            output_filename = os.path.join(input_path)
        # Read the csv into a numpy array
        data = pandas.read_csv(input_filename, usecols=self.columns).as_matrix().T
        # Extract features from the numpy array
        # Uses same variable name since data is not needed after feature extraction
        data = transform.extract(data, method="fft", data_size=chunk_size)
        # Generate an image from the resulting feature representation
        Image.fromarray(data * 255, "L").save(output_filename)
    def convert_all(self, method=None, chunk_size=1024):
        """Iterates over the training files set and generates corresponding images
        using the feature extraction method
        Keyword arguments:
        method -- Method used by extract_features to generate image data"""
        #TODO: implement this
        print("DUMMY: CONVERTING ALL CSV FILES") 
