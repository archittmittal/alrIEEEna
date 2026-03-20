# alrIEEEna26 ML Challenge by IEEE GEHU

Welcome to the dataset! This repository contains all the images and metadata you need to train your models and generate your final predictions for submissions.

## Dataset Contents

* **`images/`**: This directory contains all the `.jpg` image files for both the training and testing phases.
* **`TRAIN.csv`**: The training dataset. It contains two columns: `IMAGE` (the exact filename) and `LABEL` (the target class).
* **`TEST.csv`**: The test dataset. It contains only the `IMAGE` column. You will use your trained model to predict the missing labels for these images.

---

## Submission Guidelines

Once you have generated predictions for the images listed in `TEST.csv`, you must format your results and upload a file named **`FINAL.csv`**.

Your submission file must contain exactly two columns, spelled exactly like this:
1. **`IMAGE`**: The exact filename as it appears in the test set.
2. **`LABEL`**: Your model's predicted class label (integer) for that image.

### Sample Submission Format (`FINAL.csv`)

If your model predicts that `061857.jpg` is class 145, `061861.jpg` is class 32, and so on, your `FINAL.csv` should look exactly like this:

| IMAGE | LABEL |
| :--- | :--- |
| 061857.jpg | 145 |
| 061861.jpg | 32 |
| 061862.jpg | 296 |
| 061863.jpg | 8 |
| 061865.jpg | 112 |

> **Important Formatting Notes:** > * Ensure your output file is saved as a standard comma-separated values (`.csv`) file named 'FINAL.csv'.
> * Do not include any hidden spaces in the column headers.
> * Your submission must have the exact same number of rows as `TEST.csv`.

Good luck!