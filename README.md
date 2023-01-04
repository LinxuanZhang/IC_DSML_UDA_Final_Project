# IC_DSML_UDA_Final_Project
Scripts for IC DSML MATH70103: Unstructured Data Analysis Final Project

versions:

- Python: 3.9.12
- tensorflow: 2.9.1
- tensorflow_datasets: 4.7.0
- numpy: 1.21.5
- pandas: 1.4.2
- skimage: 0.19.2
- PIL: 9.0.1
- matplotlib: 3.5.1
- seaborn: 0.11.2

There are three scripts that can be run in the order labeled:
1. EDA: this script aims to plot some example images from training/validation/testing set, the plotted results was NOT pushed.
2. training: this script trains the models and save the model in the model dir and the training history in the history_plot dir. To save your time of retraining, the trained model was pushed. The training histories were NOT pushed.
3. analysis: this script reads in the trained model and infer on validation/testing set, it also plots some results chart for analysis. All of the plots are NOT pushed.
