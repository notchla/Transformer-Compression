#almost useless right now, just to check that the compressed version works

filename = "/cluster/home/fdeaglio/deep-learning/evaluations/42289/compressed.json"

import json
with open(filename, "r") as f:
    print(json.load(f)[0])