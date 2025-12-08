import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras
import pyreadr
import numpy as np
import pandas
import matplotlib.pyplot as plt
from Scripts.brno_cnnvae_106.m106 import VAE, Sampler


data = pyreadr.read_r("/path/to/dataset.RData")
# ------------------------------------------------------------------
#             DS2 Train Data  --------------------------------------
# ------------------------------------------------------------------
train_data = data['class.signal.train'].to_numpy()
### Max - Norm with DS1 train set max value ----------------------------------------
train_data = train_data[..., np.newaxis]
train_data /= 12955.911823647295
#-----------------------------------------------------------------------------------
train_labels = data['class.annot.train']
train_labels.columns = ['label'] # Rename the column from '0' to 'label'
train_labels = train_labels.to_numpy()
train_labels = np.squeeze(train_labels)


# ------------------------------------------------------------------
#             DS2 Test Data  ---------------------------------------
# ------------------------------------------------------------------
test_data = data['class.signal.test'].to_numpy()
### Max - Norm with DS1 train set max value -----------------------------------------
test_data = test_data[..., np.newaxis]
test_data /= 12955.911823647295
#------------------------------------------------------------------------------------
test_labels = data['class.annot.test']
test_labels.columns = ['label'] # Rename the column from '0' to 'label'
test_labels = test_labels.to_numpy()


# ------------------------------------------------------------------
#           Load Saved Model Weights  ------------------------------
# ------------------------------------------------------------------
new_vae = keras.models.load_model("noise_and_arrhythmia_identification/noise/src/vae/weights/brno_cnnvae_106/106.keras", compile=False)

# ------------------------------------------------------------------
#           Plot Windows of Each Noise Label & Reconstructions  ----
# ------------------------------------------------------------------
window_class1 = train_data[train_labels == 1, :, :]
reconstruction1 = new_vae.predict(window_class1, batch_size=128, verbose=1)
arr1 = np.random.randint(0, window_class1.shape[1], size=5)
for i in arr1:
    plt.figure(figsize=(12, 6))
    plt.plot(reconstruction1[i], label="Reconstructed Window", color="red")
    plt.plot(window_class1[i], label="Original Window", color="blue")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title("Reconstructed vs Original Noise Label 1 Window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"noise_and_arrhythmia_identification/noise/src/results/vae/ds2_noise1_plot_{i}.png")

window_class2 = train_data[train_labels == 2, :, :]
reconstruction2 = new_vae.predict(window_class2, batch_size=128, verbose=1)
arr2 = np.random.randint(0, window_class2.shape[1], size=5)
for i in arr2:
    plt.figure(figsize=(12, 6))
    plt.plot(reconstruction2[i], label="Reconstructed Window", color="red")
    plt.plot(window_class2[i], label="Original Window", color="blue")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title("Reconstructed vs Original Noise Label 2 Window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"noise_and_arrhythmia_identification/noise/src/results/vae/ds2_noise2_plot_{i}.png")

window_class3 = train_data[train_labels == 3, :, :]
reconstruction3 = new_vae.predict(window_class3, batch_size=128, verbose=1)
arr3 = np.random.randint(0, window_class3.shape[1], size=5)
for i in arr3:
    plt.figure(figsize=(12, 6))
    plt.plot(reconstruction3[i], label="Reconstructed Window", color="red")
    plt.plot(window_class3[i], label="Original Window", color="blue")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title("Reconstructed vs Original Noise Label 3 Window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"noise_and_arrhythmia_identification/noise/src/results/vae/ds2_noise3_plot_{i}.png")
