from utils import TreeDigester
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafile', help='file containing data for transforming', required=True, dest='datafile')
    parser.add_argument('-o', '--output', help='name of output file', dest='output', required=True)
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()
    tree_file = args['datafile']
    output_file = args['output']
    td = TreeDigester(tree_file)
    td.output_trees(output_file)