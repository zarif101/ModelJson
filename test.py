import json
import argparse
import torch
from ModelJson import get_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-jsonpath')
    args = parser.parse_args()
    jsonpath = args.jsonpath
    model = get_model(jsonpath)
    print(model)
    print('\n If a pytorch model summary was printed above, the test has PASSED!')

if __name__=='__main__':
    main()
