from bagpy import bagreader

import pandas as pd
import matplotlib.pyplot as plt


def read_data(path):
    '''
    Reading the bag file into a pandas DataFrame.
    Args:
        path: the path to the bag file
    '''
    # reading and making a folder with a csv file in it
    b = bagreader(path)
    b.message_by_topic('Coordinate')

    # reading the csv and returning the x and y coordinates
    df = pd.read_csv(f"{path.split('.')[0]}/oordinate.csv")
    x, y, _ = df['x'], df['y'], df['theta']

    return x, y


def plot_bag(x, y, save_path):
    '''
    Plotting the x and y coordinates of the world frame.
    Args:
        x: the x coordinate
        y: the y coordinate
    '''

    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    bag_path = '../data/robot_data.bag'
    x, y = read_data(bag_path)
    plot_bag(x, y, save_path='data/world_frame.png')
