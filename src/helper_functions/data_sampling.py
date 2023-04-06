import glob
import numpy as np
import shutil

""" Extract subsets from MS COCO dataset for training/validation/test. """

# Sample a random subset from training data for training
train_path = glob.glob("../initData/MS_COCO/training_set/train2014/*.jpg")
np.random.seed(0)  # Seeding for reproducible results
train_subset = np.random.choice(train_path, 8_000, replace=False)

# Create smaller training dataset
for filename in train_subset:
    shutil.copy(filename, "../Dataset/training/")

# Sample a random subset from validation data for validation
validation_path = glob.glob("../initData/MS_COCO/validation_set/val2014/*.jpg")
np.random.seed(0)
val_subset = np.random.choice(validation_path, 2_000, replace=False)

# Create smaller validation dataset
for filename in val_subset:
    shutil.copy(filename, "../Dataset/validation/")

# Sample a random subset from test data for testing
test_path = glob.glob("../initData/MS_COCO/test_set/test2014/*.jpg")
np.random.seed(0)
test_subset = np.random.choice(test_path, 2_000, replace=False)

# Create smaller test dataset
for filename in test_subset:
    shutil.copy(filename, "../Dataset/test/")
