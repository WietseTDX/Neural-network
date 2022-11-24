import numpy as np

def load_data_from_csv(train_data_set_file: str, validation_data_set_file: str, test_data_set_file: str):
    """The data from a csv tot seperated data sets

    Args:
        train_data_set_file (str): Train data set
        validation_data_set_file (str): Validation data set
        test_data_set_file (str): Test data set

    Returns:
        2D array, 2D array, 2D array: Train data, validation data, test data
    """    
    train_truth_table_awnsers = np.genfromtxt(train_data_set_file, delimiter=',', usecols=(-3,-2,-1),dtype=float)
    train_truth_table = np.genfromtxt(train_data_set_file, delimiter=',', usecols=(0,1,2,3), dtype=float)
    
    validation_truth_table_awnsers = np.genfromtxt(validation_data_set_file, delimiter=',', usecols=(-3,-2,-1),dtype=float)
    validation_truth_table = np.genfromtxt(validation_data_set_file, delimiter=',', usecols=(0,1,2,3), dtype=float)

    test_truth_table_awnsers = np.genfromtxt(test_data_set_file, delimiter=',', usecols=(-3,-2,-1),dtype=float)
    test_truth_table = np.genfromtxt(test_data_set_file, delimiter=',', usecols=(0,1,2,3), dtype=float)


    return (train_truth_table, train_truth_table_awnsers), (validation_truth_table, validation_truth_table_awnsers), (test_truth_table, test_truth_table_awnsers)



