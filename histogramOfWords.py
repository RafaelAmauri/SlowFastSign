import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pprint
import sys


# First we get all of the possible words in the training dataset
wordCountTraining = defaultdict(lambda: 0)

with open("./dataset/phoenix2014/phoenix-2014-multisigner/annotations/manual/train.corpus.csv", "r") as f:
    lines = f.readlines()
    del lines[0]

# For every line in the CSV
for line in lines:
    # We get the gloss (the last field) and split in the " " character
    glosses = line.split("|")[-1].split(" ")

    # For every gloss
    for g in glosses:
        # We remove the trailing newline (if there is)
        g = g.removesuffix("\n")

        # And add it to the words dict with a frequency of 0.
        wordCountTraining[g] += 1


# Sort the words alphabetically
wordCountTraining = dict(sorted(wordCountTraining.items(), key=lambda x:x[0]))


# Open the CSV with the chosen samples
selectedWordsFile = str(sys.argv[1])
with open(selectedWordsFile, "r") as f:
    lines = f.readlines()
    del lines[0]


numWords = 0
wordCountChosenCSV = defaultdict(lambda: 0)
# For every line in the CSV
for line in lines:
    # We AGAIN get the gloss (the last field) and split in the " " character
    glosses = line.split("|")[-1].split(" ")

    # For every gloss
    for g in glosses:
        # We remove the trailing newline (if there is)
        g = g.removesuffix("\n")

        if g == "":
            continue
        if g not in wordCountTraining:
            print(f"{g} not in Training set words!")

        # And add it to the words dict with a frequency of 0.
        wordCountChosenCSV[g] += 1
        numWords += 1


print(f"{numWords} total Words, {numWords / len(lines):.4f} words / video")


top100 = sorted(wordCountChosenCSV.items(), key=lambda x: x[1], reverse=True)[ : 75]

words_list = []
counts     = []

for w, c in top100:
    words_list.append(w)
    counts.append(c)

print("Top 20 most picked words:")
for w, c in top100[ : 20]:
    pprint.pprint(f"{w}: {c} ({c / wordCountTraining[w]:.4f} %)")


plt.figure(figsize=(20, 6))
plt.scatter(words_list, counts)

plt.yticks(np.arange(0, 301, step=15))
plt.xticks(rotation=90, fontsize=12)  # Remove word name for more readability
plt.title("Word Frequency (alphabetical)")
plt.xlabel("Word ID")
plt.ylabel("Occurences")
plt.tight_layout()
plt.show()