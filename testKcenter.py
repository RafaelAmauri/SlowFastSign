from active_learning_modules.rankByFeature import rankSimiliratyByFeatures, readFeaturesFromFile


# Read the extracted features for all samples in the labeled and unlabeled sets
featuresLabeledSet, featuresUnlabeledSet = readFeaturesFromFile("test/phoenix2014Signer1-test-labeled-run1-features/train", 
                                                                "test/phoenix2014Signer1-test-unlabeled-run1-features/test")


# Use the features to find the data points in the unlabeled set that are the most dissimilar to the ones in the labeled set.
mostInformativeSamples = rankSimiliratyByFeatures(  featuresLabeledSet, 
                                                    featuresUnlabeledSet,
                                                    "kcenter",
                                                    f"./",
                                                    )