# Pytorch_ReNN
Implementation of a Recursive Neural Network in Pytorch

Files are detailed as follows:
- model.py
  - Pytorch implementation of a Recursive Neural Network
- main.py
  - Main training file. Usage is: `python main.py`, run `python main.py --help` for a list of arguments.
- tree.py
  - Objects and functions for tree processing
- utils.py
  - Utility classes/functions to process the Stanford Sentiment Treebank (SSTB)
- transform_parse_trees.py
  - Script to transform the SSTB into accepted format
- trees/
  - Directory containing preprocessed trees in accepted format
