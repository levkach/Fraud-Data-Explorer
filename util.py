import os

colorscale = [[0.0, 'rgb(0,51,255)'],
              [0.2, 'rgb(255,155,84)'], [0.4, 'rgb(255,127,81)'],
              [0.6, 'rgb(206,66,87)'],
              [0.8, 'rgb(114,0,38)'],
              [1.0, 'rgb(79,0,11)'],
              ]


# Finds csv files
def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]
