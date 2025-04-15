import matplotlib.pyplot as plt
import numpy as np
import random

from active_learning_modules.rankByFeature import rankSimiliratyByFeatures



xAxisLimit = 400
yAxisLimit = 400

nLabeledSamples    = 1
nUnlabeledSamples  = 5
nFeatureDimensions = 2
nFrames            = 1
nLabelings         = 1
strategy           = "kcenter" # Can be "cosine" or "kcenter"


# Generate a bunch of features for the labeled and unlabeled sets
#labeledFeatures   = np.asarray([[[random.randint(1, xAxisLimit) for _ in range(nFeatureDimensions)] for _ in range(nFrames)] for _ in range(nLabeledSamples)])
labeledFeatures   = np.asarray([[[200,200]]])
unlabeledFeatures = np.asarray([[[random.randint(1, xAxisLimit) for _ in range(nFeatureDimensions)] for _ in range(nFrames)] for _ in range(nUnlabeledSamples)])

unlabeledFeatures = np.concatenate([unlabeledFeatures, [[[400,400]]]], axis=0)
nUnlabeledSamples+= 1

# This converts the labeled and unlabeledFeatures arrays from the shape

# (nLabeledSamples  , 1, nFeatureDimensions) to (nLabeledSamples  , nFeatureDimensions)
# (nUnlabeledSamples, 1, nFeatureDimensions) to (nUnlabeledSamples, nFeatureDimensions)
if nFrames == 10:
    labeledFeatures   = labeledFeatures.reshape(nLabeledSamples    , -1)
    unlabeledFeatures = unlabeledFeatures.reshape(nUnlabeledSamples, -1)


# Now we "fake" a call to active_learning_modules.rankbyfeature.readFeaturesFromFile
# This function would return our features as a dict, where the keys are the names of the videos and the values
# are the features.
featuresLabeledSet   = { f"/tmp/navegador/{i}" : labeledFeatures[i]                              for i in range(nLabeledSamples)                                                                }
featuresUnlabeledSet = { f"/tmp/navegador/{i}" : unlabeledFeatures[i] for i in range(nUnlabeledSamples) }#if unlabeledConfidences[i] < np.median(unlabeledConfidences) }

fig, axs = plt.subplots(2, 2, figsize=(12, 6))


similarityRank       = rankSimiliratyByFeatures(featuresLabeledSet, 
                                                featuresUnlabeledSet,
                                                strategy,
                                                "/tmp/navegador/",
                                                )

print(similarityRank)


# Plot the unlabeled samples and their corresponding confidences
sc1 = axs[0][0].scatter(unlabeledFeatures[..., 0], unlabeledFeatures[..., 1], s=100, marker='s')
axs[0][0].set_title('Unlabeled Set')
axs[0][0].set_xlabel('Feature 1')
axs[0][0].set_ylabel('Feature 2')
axs[0][0].set_xlim(0, xAxisLimit)
axs[0][0].set_ylim(0, yAxisLimit)


# Plot the labeled samples
sc2 = axs[0][1].scatter(labeledFeatures[..., 0], labeledFeatures[..., 1], s=100, marker='s')
axs[0][1].set_title('Original Labeled Set')
axs[0][1].set_xlabel('Feature 1')
axs[0][1].set_ylabel('Feature 2')
axs[0][1].set_xlim(0, xAxisLimit)
axs[0][1].set_ylim(0, yAxisLimit)


# Plot the selected samples
addedSamples = 0
for idx, conf in similarityRank.items():
    if addedSamples < nLabelings:
        idx = f"/tmp/navegador/{idx}"

        # Extract the features and confidence
        currentSelectedSampleFeature = featuresUnlabeledSet[idx]
        
        # Append it to the labeled set (will be used for plotting the new labeled set)
        labeledFeatures = np.concatenate([labeledFeatures, np.expand_dims(currentSelectedSampleFeature, axis=0)], axis=0)
            
        # Plot the current sample on a scatter plot
        sc3 = axs[1][0].scatter(currentSelectedSampleFeature[..., 0], currentSelectedSampleFeature[..., 1], s=100, marker='s')
        addedSamples += 1



axs[1][0].set_title('Chosen Samples')
axs[1][0].set_xlabel('Feature 1')
axs[1][0].set_ylabel('Feature 2')
axs[1][0].set_xlim(0, xAxisLimit)
axs[1][0].set_ylim(0, yAxisLimit)


# Plot the new labeled set
sc4 = axs[1][1].scatter(labeledFeatures[..., 0], labeledFeatures[..., 1], s=100, marker='s')
axs[1][1].set_title('New Labeled Set')
axs[1][1].set_xlabel('Feature 1')
axs[1][1].set_ylabel('Feature 2')
axs[1][1].set_xlim(0, xAxisLimit)
axs[1][1].set_ylim(0, yAxisLimit)

plt.tight_layout()
plt.show()