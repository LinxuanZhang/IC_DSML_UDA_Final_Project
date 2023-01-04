import numpy as np
import tensorflow as tf
import pandas as pd
from itertools import combinations
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# setting random seed
tf.random.set_seed(123)

# constants
IMG_SIZE = 28
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE
test_dir = 'test'
augs = ['flip', 'rotate', 'bright', 'contrast', 'translate', 'zoom']

# laod training and validation set from tfds
(train_ds, val_ds), dataset_info = tfds.load(
    'fashion_mnist',
    split=['train[:70%]', 'train[70%:]'],
    with_info=True,
    as_supervised=True,
)

# load test set from directory
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    color_mode='grayscale',
    image_size=(IMG_SIZE, IMG_SIZE))


# resize layer to makes sure all training/validation/test data are of the same size
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255)
])


# define a prepare function to prepare validation and test set
@tf.autograph.experimental.do_not_convert
def prepare(ds):
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds.prefetch(buffer_size=AUTOTUNE)


val_ds = prepare(val_ds)

# get validation/testing results from the model with no data augmentation
model = tf.keras.models.load_model('model/no_aug')
val_acc_oiriginal = model.evaluate(val_ds)[1]
test_acc_original = model.evaluate(test_ds)[1]

# get validation/testing results from models with 1 data augmentation applied
val_acc1 = [val_acc_oiriginal]
test_acc1 = [test_acc_original]
for aug in augs:
    model = tf.keras.models.load_model(f'model/1/{aug}')
    val_acc1.append(model.evaluate(val_ds)[1])
    test_acc1.append(model.evaluate(test_ds)[1])

# barh plot comparing validation accuracies of models with 1 data augmentation applied
val_acc1_df = pd.DataFrame({'augmentation':['no augmentation'] + augs, 'validation accuracy':val_acc1}).sort_values('validation accuracy')
val_acc1_df.plot.barh(x='augmentation', y='validation accuracy')
plt.show()
plt.clf()

# barh plot comparing test accuracies of models with 1 data augmentation applied
test_acc1_df = pd.DataFrame({'augmentation':['no augmentation'] + augs, 'test accuracy':test_acc1}).sort_values('test accuracy')
test_acc1_df.plot.barh(x='augmentation', y='test accuracy')
plt.show()
plt.clf()

# get validation/testing results from models with 2 data augmentation applied
val_acc2 = []
augs2 = []
test_acc2 = []
for aug in combinations(augs,2):
    model = tf.keras.models.load_model(f'model/2/{"_".join(aug)}')
    val_acc2.append(model.evaluate(val_ds)[1])
    test_acc2.append(model.evaluate(test_ds)[1])
    augs2.append(aug)


# plot heatmap to compare validation/test accuracies of models with 2 data augmentation applied
val_acc2_df_heatmap = pd.DataFrame(index=augs, columns=augs)
test_acc2_df_heatmap = pd.DataFrame(index=augs, columns=augs)
for i, aug in enumerate(augs2):
    val_acc2_df_heatmap.loc[aug[0], aug[1]] = val_acc2[i]
    val_acc2_df_heatmap.loc[aug[1], aug[0]] = val_acc2[i]
    test_acc2_df_heatmap.loc[aug[0], aug[1]] = test_acc2[i]
    test_acc2_df_heatmap.loc[aug[1], aug[0]] = test_acc2[i]

val_acc2_df_heatmap = val_acc2_df_heatmap.astype(float)
sns.heatmap(val_acc2_df_heatmap, annot=True)
plt.title('validation accuracy')
plt.show()
plt.clf()

test_acc2_df_heatmap = test_acc2_df_heatmap.astype(float)
sns.heatmap(test_acc2_df_heatmap, annot=True)
plt.title('test accuracy')
plt.show()
plt.clf()

# get validation/testing results from models with 3 data augmentation applied
val_acc3 = []
augs3 = []
test_acc3 = []
for aug in combinations(augs,3):
    model = tf.keras.models.load_model(f'model/3/{"_".join(aug)}')
    val_acc3.append(model.evaluate(val_ds)[1])
    test_acc3.append(model.evaluate(test_ds)[1])
    augs3.append(aug)

# get validation/testing results from models with 4 data augmentation applied
val_acc4 = []
augs4 = []
test_acc4 = []
for aug in combinations(augs,4):
    model = tf.keras.models.load_model(f'model/4/{"_".join(aug)}')
    val_acc4.append(model.evaluate(val_ds)[1])
    test_acc4.append(model.evaluate(test_ds)[1])
    augs4.append(aug)

# get validation/testing results from models with 5 data augmentation applied
val_acc5 = []
augs5 = []
test_acc5 = []
for aug in combinations(augs,5):
    model = tf.keras.models.load_model(f'model/5/{"_".join(aug)}')
    val_acc5.append(model.evaluate(val_ds)[1])
    test_acc5.append(model.evaluate(test_ds)[1])
    augs5.append(aug)

# get validation/testing results from models with 6 data augmentation applied
val_acc6 = []
augs6 = []
test_acc6 = []
for aug in combinations(augs,6):
    model = tf.keras.models.load_model(f'model/6/{"_".join(aug)}')
    val_acc6.append(model.evaluate(val_ds)[1])
    test_acc6.append(model.evaluate(test_ds)[1])
    augs6.append(aug)

cols = ['number of augmentations', 'average validation accuracy', 'max validation accuracy', 'average test accuracy', 'max test accuracy']
summary_tab = pd.DataFrame(columns=cols)
summary_tab = summary_tab.append(pd.DataFrame([[0, np.mean(val_acc_oiriginal), np.max(val_acc_oiriginal), np.mean(test_acc_original), np.max(test_acc_original)]], columns=cols), ignore_index=True)
summary_tab = summary_tab.append(pd.DataFrame([[1, np.mean(val_acc1), np.max(val_acc1), np.mean(test_acc1), np.max(test_acc1)]], columns=cols), ignore_index=True)
summary_tab = summary_tab.append(pd.DataFrame([[2, np.mean(val_acc2), np.max(val_acc2), np.mean(test_acc2), np.max(test_acc2)]], columns=cols), ignore_index=True)
summary_tab = summary_tab.append(pd.DataFrame([[3, np.mean(val_acc3), np.max(val_acc3), np.mean(test_acc3), np.max(test_acc3)]], columns=cols), ignore_index=True)
summary_tab = summary_tab.append(pd.DataFrame([[4, np.mean(val_acc4), np.max(val_acc4), np.mean(test_acc4), np.max(test_acc4)]], columns=cols), ignore_index=True)
summary_tab = summary_tab.append(pd.DataFrame([[5, np.mean(val_acc5), np.max(val_acc5), np.mean(test_acc5), np.max(test_acc5)]], columns=cols), ignore_index=True)
summary_tab = summary_tab.append(pd.DataFrame([[6, np.mean(val_acc6), np.max(val_acc6), np.mean(test_acc6), np.max(test_acc6)]], columns=cols), ignore_index=True)

val_acc1_wo_bright = [val_acc1[i] for i in range(len(augs)) if 'bright' not in augs[i]]
val_acc2_wo_bright = [val_acc2[i] for i in range(len(augs2)) if 'bright' not in augs2[i]]
val_acc3_wo_bright = [val_acc3[i] for i in range(len(augs3)) if 'bright' not in augs3[i]]
val_acc4_wo_bright = [val_acc4[i] for i in range(len(augs4)) if 'bright' not in augs4[i]]
val_acc5_wo_bright = [val_acc5[i] for i in range(len(augs5)) if 'bright' not in augs5[i]]

val_acc1_wi_bright = [val_acc1[i] for i in range(len(augs)) if 'bright' in augs[i]]
val_acc2_wi_bright = [val_acc2[i] for i in range(len(augs2)) if 'bright' in augs2[i]]
val_acc3_wi_bright = [val_acc3[i] for i in range(len(augs3)) if 'bright' in augs3[i]]
val_acc4_wi_bright = [val_acc4[i] for i in range(len(augs4)) if 'bright' in augs4[i]]
val_acc5_wi_bright = [val_acc5[i] for i in range(len(augs5)) if 'bright' in augs5[i]]

test_acc1_wo_bright = [test_acc1[i] for i in range(len(augs)) if 'bright' not in augs[i]]
test_acc2_wo_bright = [test_acc2[i] for i in range(len(augs2)) if 'bright' not in augs2[i]]
test_acc3_wo_bright = [test_acc3[i] for i in range(len(augs3)) if 'bright' not in augs3[i]]
test_acc4_wo_bright = [test_acc4[i] for i in range(len(augs4)) if 'bright' not in augs4[i]]
test_acc5_wo_bright = [test_acc5[i] for i in range(len(augs5)) if 'bright' not in augs5[i]]

test_acc1_wi_bright = [test_acc1[i] for i in range(len(augs)) if 'bright' in augs[i]]
test_acc2_wi_bright = [test_acc2[i] for i in range(len(augs2)) if 'bright' in augs2[i]]
test_acc3_wi_bright = [test_acc3[i] for i in range(len(augs3)) if 'bright' in augs3[i]]
test_acc4_wi_bright = [test_acc4[i] for i in range(len(augs4)) if 'bright' in augs4[i]]
test_acc5_wi_bright = [test_acc5[i] for i in range(len(augs5)) if 'bright' in augs5[i]]


vo = np.mean(val_acc1_wo_bright + val_acc2_wo_bright + val_acc3_wo_bright + val_acc4_wo_bright + val_acc5_wo_bright)
vi = np.mean(val_acc1_wi_bright + val_acc2_wi_bright + val_acc3_wi_bright + val_acc4_wi_bright + val_acc5_wi_bright)

to = np.mean(test_acc1_wo_bright + test_acc2_wo_bright + test_acc3_wo_bright + test_acc4_wo_bright + test_acc5_wo_bright)
ti = np.mean(test_acc1_wi_bright + test_acc2_wi_bright + test_acc3_wi_bright + test_acc4_wi_bright + test_acc5_wi_bright)

print(f'Average validation accuracy without brightness augmentation is {vo}')
print(f'Average validation accuracy with combination of brightness augmentation is {vi}')
print(f'Average test accuracy without brightness augmentation is {to}')
print(f'Average test accuracy with combination brightness augmentation is {ti}')

summary_tab.sort_values('number of augmentations', ascending=False).plot.barh(x='number of augmentations', y=['average validation accuracy', 'max validation accuracy'])
plt.title('Validation Accuracies')
plt.show()
plt.clf()

summary_tab.sort_values('number of augmentations', ascending=False).plot.barh(x='number of augmentations', y=['average test accuracy', 'max test accuracy'])
plt.title('Test Accuracies')
plt.show()
plt.clf()

summary_tab_wob = pd.DataFrame(columns=cols)
summary_tab_wob = summary_tab_wob.append(pd.DataFrame([[1, np.mean(val_acc1_wo_bright), np.max(val_acc1_wo_bright), np.mean(test_acc1_wo_bright), np.max(test_acc1_wo_bright)]], columns=cols), ignore_index=True)
summary_tab_wob = summary_tab_wob.append(pd.DataFrame([[2, np.mean(val_acc2_wo_bright), np.max(val_acc2_wo_bright), np.mean(test_acc2_wo_bright), np.max(test_acc2_wo_bright)]], columns=cols), ignore_index=True)
summary_tab_wob = summary_tab_wob.append(pd.DataFrame([[3, np.mean(val_acc3_wo_bright), np.max(val_acc3_wo_bright), np.mean(test_acc3_wo_bright), np.max(test_acc3_wo_bright)]], columns=cols), ignore_index=True)
summary_tab_wob = summary_tab_wob.append(pd.DataFrame([[4, np.mean(val_acc4_wo_bright), np.max(val_acc4_wo_bright), np.mean(test_acc4_wo_bright), np.max(test_acc4_wo_bright)]], columns=cols), ignore_index=True)
summary_tab_wob = summary_tab_wob.append(pd.DataFrame([[5, np.mean(val_acc5_wo_bright), np.max(val_acc5_wo_bright), np.mean(test_acc5_wo_bright), np.max(test_acc5_wo_bright)]], columns=cols), ignore_index=True)

summary_tab_wob.sort_values('number of augmentations', ascending=False).plot.barh(x='number of augmentations', y=['average validation accuracy', 'max validation accuracy'])
plt.title('Validation Accuracies without brightness augmentation')
plt.show()
plt.clf()
