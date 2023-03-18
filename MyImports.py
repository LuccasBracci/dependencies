class MyImports:
    def __init__(self):
        self.pd = None
        self.np = None
        self.plt = None 
        self.train_test_split = None
        self.make_column_selector = None
        self.make_column_transformer = None
        self.StandardScaler = None
        self.OneHotEncoder = None
        self.make_pipeline = None
        self.DummyRegressor = None
        self.DecisionTreeRegressor = None
        self.RandomForestRegressor = None
        self.r2_score = None
        self.mean_absolute_error = None
        self.mean_squared_error = None
        self.set_config = None
        self.clear_output = None

    def import_all(self):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.compose import make_column_selector, make_column_transformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.pipeline import make_pipeline
        from sklearn.dummy import DummyRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        from sklearn import set_config
        from IPython.core.display import clear_output

        self.pd = pd
        self.np = np
        self.plt = plt
        self.train_test_split = train_test_split
        self.make_column_selector = make_column_selector
        self.make_column_transformer = make_column_transformer
        self.StandardScaler = StandardScaler
        self.OneHotEncoder = OneHotEncoder
        self.make_pipeline = make_pipeline
        self.DummyRegressor = DummyRegressor
        self.DecisionTreeRegressor = DecisionTreeRegressor
        self.RandomForestRegressor = RandomForestRegressor
        self.r2_score = r2_score
        self.mean_absolute_error = mean_absolute_error
        self.mean_squared_error = mean_squared_error
        self.set_config = set_config
        self.clear_output = clear_output
