import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--concat", action="store_true")
args = parser.parse_args()

print(args.concat)
