import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('banks', nargs='*', help='List of bank No.s')

args = parser.parse_args()

print("Bank IDs:", args.banks)

HAMMER_ROOT = os.environ['HAMMER_ROOT']