import numpy
import re
import scipy
from scipy.spatial import distance

sentences = open("sentenses.txt", "r").read().splitlines()
words = []
uniqWords = []

for line in sentences:
    for word in re.split('[^a-z]', line.lower()):
        if word: 
            words.append(word)
            if not word in uniqWords:
                uniqWords.append(word)

allWords = [None] * len(words)

for word in uniqWords:
    allWords[words.index(word)] = word

wordMat = numpy.zeros(shape=(len(sentences),len(allWords)))

for i in range(len(sentences)):
    for j in range(len(allWords)):
    	if allWords[j]:
            wordMat[i, j] = sentences[i].count(allWords[j])
        
distances = {}

for i in range(len(sentences)):
    distances[i] = scipy.spatial.distance.cosine(wordMat[0], wordMat[i])

orderedDistances = sorted(distances.items(), key=lambda x:x[1])

print orderedDistances[1:4]