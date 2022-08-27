import pandas as pd
from sklearn.model_selection import train_test_split


class BaseDataLoaderVarChecker(object):
    def _determine_column_exists(self, indicator_column, columns):
        return indicator_column in columns

    def _determine_supervised(self, indicator_column):
        return False if indicator_column is None else True

class BaseDataLoader(BaseDataLoaderVarChecker):
    def __init__(self, 
                    filepath: str, 
                    columns_to_keep: list=None, 
                    indicator_column: str=None,
                    split_data: bool=False,
                    test_ratio: float=0.5,
                    **kwargs):
        '''
        BaseDataLoader class designed to read a csv and split the data based on user selection
        
        
        '''
        BaseDataLoaderVarChecker().__init__()

        self.df = None
        self.filepath = filepath

        self.columns_to_keep = columns_to_keep
        self.indicator_column = indicator_column
        self.split_data = split_data
        self.test_ratio = test_ratio
        self.kwargs = kwargs
        
    def get_data(self):
        self._read_csv(self.filepath, self.kwargs)
        if self._determine_supervised(self.indicator_column):
            ''' data is Supervised'''

            # Data is supervised so we need to include indicator column
            if self.indicator_column is not None:
                self.columns_to_keep.append(self.indicator_column)
            
            X, y = self._prepare_data_for_supervised(self.columns_to_keep)
            if self.split_data:
                return self._split_data_to_train_test(X, y, test_ratio=self.test_ratio, **self.kwargs)
            else: 
                return X, y
        else:
            ''' data is unsupervised'''
            return self.df[self.columns_to_keep]

    def _read_csv(self, filepath, kwargs):
        self.df = pd.read_csv(filepath, **kwargs)

    def _split_data_to_train_test(self, X, y, test_ratio, kwargs):
        return train_test_split(X, y, test_ratio=test_ratio, **kwargs)

    def _split_dataframe_to_x_y(self, columns_to_keep):
        return self.df[columns_to_keep[:-1]],  self.df[[columns_to_keep[-1]]]

    def _prepare_data_for_supervised(self, columns_to_keep):
        return self._split_dataframe_to_x_y(columns_to_keep)