import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from skimage import filters
from PIL import Image
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model,Input

# setting random seed
tf.random.set_seed(123)

# laod training and validation set from tfds
(train_ds, val_ds), dataset_info = tfds.load(
    'fashion_mnist',
    split=['train[:70%]', 'train[70%:]'],
    with_info=True,
    as_supervised=True,
)

# get one example from the training set and plot
x, y = next(iter(train_ds))
x = x.numpy()
plt.imshow(x, cmap='Greys')
plt.savefig('original example')
plt.clf()


flip = layers.RandomFlip("horizontal_and_vertical")(x, training=True).numpy()
flip = flip.astype('uint8')
rotate = layers.RandomRotatpltn(0.4)(x, training=True).numpy()
rotate = rotate.astype('uint8')
bright = layers.RandomBrightness(0.2)(x, training=True).numpy()
bright = bright.astype('uint8')
contrast = layers.RandomContrast(0.2)(x, training=True).numpy()
contrast = contrast.astype('uint8')
translate = layers.RandomTranslatpltn(0.2, 0.2)(x, training=True).numpy()
translate = translate.astype('uint8')
zoom = layers.RandomZoom(0.2)(x, training=True).numpy()
zoom = zoom.astype('uint8')

# plot example of
figure, axis = plt.subplots(2, 3)
axis[0, 0].imshow(flip, cmap='Greys')
axis[0, 0].set_title("flip")
axis[0, 1].imshow(rotate, cmap='Greys')
axis[0, 1].set_title("rotate")
axis[0, 2].imshow(bright, cmap='Greys')
axis[0, 2].set_title("bright")
axis[1, 0].imshow(contrast, cmap='Greys')
axis[1, 0].set_title("contrast")
axis[1, 1].imshow(translate, cmap='Greys')
axis[1, 1].set_title("translate")
axis[1, 2].imshow(zoom, cmap='Greys')
axis[1, 2].set_title("zoom")
plt.savefig('augs')

plt.show()
plt.clf()

# apply prewitt filter to a image from each of the 10 label
for i in range(10):
    x, y = next(iter(train_ds.filter(lambda img, label: label == i)))
    x = x.numpy()
    plt.imshow(x, cmap='Greys')
    plt.savefig(f'{y.numpy()}')
    plt.clf()
    edge = filters.prewitt(x.reshape(28, 28))
    plt.imshow(edge, cmap='Greys')
    plt.savefig(f'{y.numpy()} edge')
    plt.clf()

# plot test image example
img = Image.open(f'test/9/9o.png')
filters.prewitt(x.reshape(28, 28))
img_edge = filters.prewitt(np.array(img))
plt.imshow(img_edge, cmap='Greys')
plt.savefig(f'test edge')
plt.clf()
plt.imshow(img, cmap='Greys')
plt.savefig(f'test example')
plt.clf()


# tf.keras.utils.plot_model to visualize the model
class MyModel(Model):
    def __init__(self, dim):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(16, 3, activation='relu')
        self.conv2 = Conv2D(32, 3, activation='relu')
        self.conv3 = Conv2D(64, 3, activation='relu')
        self.mp1 = MaxPooling2D()
        self.mp2 = MaxPooling2D()
        self.mp3 = MaxPooling2D()
        self.flatten = Flatten()
        self.d1 = Dense(32, activation='relu')
        self.d2 = Dense(10)
    def call(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.mp3(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
    def build_graph(self):
        x = Input(shape=(dim))
        return Model(inputs=[x], outputs=self.call(x))

dim = (28, 28, 1)
# Create an instance of the model
model = MyModel((dim))
model.build((None, *dim))
model.build_graph().summary()

tf.keras.utils.plot_model(model.build_graph(), to_file="model.png",
                          expand_nested=True, show_shapes=True)
