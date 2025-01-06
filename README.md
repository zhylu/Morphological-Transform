These are the source codes for "Morphological Augmentation of Image Data using Three-Dimensional Discrete Fuzzy Numbers: A Theoretical Foundation".
All the involved datasets are public and can be obtained according to the reffered papers. In the practice, the datasets are pre-processed as MAT files for accelerating training. A sample of the pre-processed dataset files is provided in " pre-processed dataset.txt".
"gdfn-illustration.mlx" is an implementation for Matlab to process single image.
"gdfn.py" is an implementation in Python that can be imported for practical model training in Pytorch.
"training_baseline.py" is an example of the model training without Morphological Augmentation.
"training_gdfn.py" is an example of the model training with Morphological Augmentation.
