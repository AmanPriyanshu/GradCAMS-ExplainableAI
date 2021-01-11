import torch
import numpy as np
import pandas as pd

class SimpleCNN:
	def __init__(self):
		self.model = self.initial_model()

	def initialize_model(self):
		
