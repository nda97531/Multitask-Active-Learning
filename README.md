## Multitask-Active-Learning

### 1. Repo utilization
This is implementation of *Multi-task Active Learning for Sensor-based Human Activity Recognition*.</br>
All packages (network architecture, augmentation, trainer, ...) are in the root package `MAT`.</br>
**Running examples can be found in `jupyter`:**
- [Active source data selection](jupyter/1_train_source_dataset.ipynb)
- [Multi-task Transfer learning on target data](jupyter/2_train_MAT.ipynb)

Please install required packages first by running:</br>
`pip install -r requirements.txt`

### 2. Datasets
Four datasets used in our research:
- Source dataset: [MobiAct v2](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/) </br>
- Target datasets:
  - [MotionSense](https://github.com/mmalekzadeh/motion-sense/tree/master/data)
  - [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php)
  - [Physical Activity Recognition](https://www.utwente.nl/en/eemcs/ps/research/dataset/)
