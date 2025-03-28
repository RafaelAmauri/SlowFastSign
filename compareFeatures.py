import numpy as np
import matplotlib.pyplot as plt
import random

import numpy as np
import sklearn
import sklearn.cluster

from active_learning_modules.rankbyfeature import rankSimiliratyByFeatures



xAxisLimit = 400
yAxisLimit = 400

nLabeledSamples    = 5
nUnlabeledSamples  = 2000
nFeatureDimensions = 2
nFrames            = 1
nLabelings         = 60
strategy           = "cosine" # Can be "cosine" or "kcenter"


# Generate a bunch of unlabeled Confidences
unlabeledConfidences = np.asarray([ random.random() for i in range(nUnlabeledSamples)])

# Generate a bunch of features for the labeled and unlabeled sets
labeledFeatures   = np.asarray([[[random.randint(1, xAxisLimit) for _ in range(nFeatureDimensions)] for _ in range(nFrames)] for _ in range(nLabeledSamples)])
unlabeledFeatures = np.asarray([[[random.randint(1, xAxisLimit) for _ in range(nFeatureDimensions)] for _ in range(nFrames)] for _ in range(nUnlabeledSamples)])

if strategy == "kcenter":
    labeledFeatures   = np.mean(labeledFeatures  , axis=1)
    unlabeledFeatures = np.mean(unlabeledFeatures, axis=1)


# Now we "fake" a call to active_learning_modules.rankbyfeature.readFeaturesFromFile
# This function would return our features as a dict, where the keys are the names of the videos and the values
# are the features.
featuresLabeledSet   = { f"/tmp/navegador/{i}" : labeledFeatures[i]                              for i in range(nLabeledSamples)                                                                }
featuresUnlabeledSet = { f"/tmp/navegador/{i}" : [unlabeledFeatures[i], unlabeledConfidences[i]] for i in range(nUnlabeledSamples) if unlabeledConfidences[i] < np.median(unlabeledConfidences) }



fig, axs = plt.subplots(2, 2, figsize=(12, 6))


similarity           = rankSimiliratyByFeatures(featuresLabeledSet, 
                                                featuresUnlabeledSet,
                                                strategy,
                                                "/tmp/navegador/",
                                                nLabelings)



sc1 = axs[0][0].scatter(unlabeledFeatures[..., 0], unlabeledFeatures[..., 1], c=unlabeledConfidences, vmin=0, vmax=1, cmap='cool', s=100, marker='s')
axs[0][0].set_title('Unlabeled Set')
axs[0][0].set_xlabel('Feature 1')
axs[0][0].set_ylabel('Feature 2')
axs[0][0].set_xlim(0, xAxisLimit)
axs[0][0].set_ylim(0, yAxisLimit)
cbar1 = plt.colorbar(sc1, ax=axs[0][0])
cbar1.set_label('Confidence')


sc2 = axs[0][1].scatter(labeledFeatures[..., 0], labeledFeatures[..., 1], s=100, marker='s')
axs[0][1].set_title('Original Labeled Set')
axs[0][1].set_xlabel('Feature 1')
axs[0][1].set_ylabel('Feature 2')
axs[0][1].set_xlim(0, xAxisLimit)
axs[0][1].set_ylim(0, yAxisLimit)


addedSamples = 0
for idx, conf in similarity.items():
    if addedSamples < nLabelings:
        idx = f"/tmp/navegador/{idx}"
        
        if strategy == "kcenter":
            x    = featuresUnlabeledSet[idx][0][0]
            y    = featuresUnlabeledSet[idx][0][1]
            conf = featuresUnlabeledSet[idx][1]
            xy   = np.asarray([x, y])
            xy   = np.expand_dims(xy, axis=0)
            labeledFeatures = np.concatenate([labeledFeatures, xy], axis=0)

        else:
            x    = featuresUnlabeledSet[idx][0][0][0]
            y    = featuresUnlabeledSet[idx][0][0][1]
            conf = featuresUnlabeledSet[idx][1]
            xy   = np.asarray([x, y])
            xy   = np.expand_dims(xy, axis=0)
            labeledFeatures = np.concatenate([labeledFeatures, [xy]], axis=0)
        
        
        sc3 = axs[1][0].scatter(x, y, c=conf, vmin=0, vmax=1, cmap='cool', s=100, marker='s')
        addedSamples += 1


axs[1][0].set_title('Chosen Samples')
axs[1][0].set_xlabel('Feature 1')
axs[1][0].set_ylabel('Feature 2')
axs[1][0].set_xlim(0, xAxisLimit)
axs[1][0].set_ylim(0, yAxisLimit)
cbar3 = plt.colorbar(sc3, ax=axs[1][0])
cbar3.set_label('Confidence')


sc4 = axs[1][1].scatter(labeledFeatures[..., 0], labeledFeatures[..., 1], s=100, marker='s')
axs[1][1].set_title('New Labeled Set')
axs[1][1].set_xlabel('Feature 1')
axs[1][1].set_ylabel('Feature 2')
axs[1][1].set_xlim(0, xAxisLimit)
axs[1][1].set_ylim(0, yAxisLimit)

plt.tight_layout()
plt.show()