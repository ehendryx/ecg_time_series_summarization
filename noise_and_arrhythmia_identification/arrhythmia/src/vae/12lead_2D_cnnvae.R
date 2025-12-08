# CNN VAE for all 12 leads
setwd("D:/Documents/UCO/Research/R Scripts")
#reticulate::use_python('C:/Users/ethan/OneDrive/Documents/.virtualenvs/r-reticulate/Scripts/python.exe', required = TRUE)

#install.packages("keras")    #Installs just the Keras package to R
#keras::install_keras()       #Fully installs Keras & TF into default directory
library(keras)
library(tensorflow)
library(reticulate)
#install.packages("dplyr")
library(dplyr)



load("D:/Documents/UCO/Research/Data Sets/INCART12/combined_data.RData")
ds1.train.x = unname(ds1.train.x)
ds1.test.x = unname(ds1.test.x)
ds2.train.x = unname(ds2.train.x)
ds2.test.x = unname(ds2.test.x)

# Adding a channels dimension to the data
ds1.train.x = array_reshape(ds1.train.x, c(dim(ds1.train.x), 1))
# Adding a channels dimension to the data
ds1.test.x = array_reshape(ds1.test.x, c(dim(ds1.test.x), 1))


# FIXME: add comments to these two code blocks ---------------------------------
ds1.train.x.reshaped = array(dim = c(28166, 12, 150))
for (row in 1:nrow(ds1.train.x)){
  beat1 = ds1.train.x[row,,]
  beat1.transposed = t(beat1)
  ds1.train.x.reshaped[row,,] = beat1.transposed
}
# Adding a channels dimension to the data
ds1.train.x.reshaped = array_reshape(ds1.train.x.reshaped, c(dim(ds1.train.x.reshaped), 1))

ds1.test.x.reshaped = array(dim = c(31943, 12, 150))
for (row in 1:nrow(ds1.test.x)){
  beat1 = ds1.test.x[row,,]
  beat1.transposed = t(beat1)
  ds1.test.x.reshaped[row,,] = beat1.transposed
}
# Adding a channels dimension to the data
ds1.test.x.reshaped = array_reshape(ds1.test.x.reshaped, c(dim(ds1.test.x.reshaped), 1))



# ------------------------------------------------------------------------------
# Variables for conv layer filter amounts and for latent dimensions

intermediate_dim1 <- 1
intermediate_dim2 <- 16
intermediate_dim3 <- 32
latent_dim <- 50
img_dim <- c(150, 12)
channels = 1


tensorflow::set_random_seed(3)



#  Inputs and encoder -----------------------------------------------------

encoder_inputs <- layer_input(shape = c(img_dim, channels))

x <- encoder_inputs %>%
  # Hidden layer 1
  layer_conv_2d(intermediate_dim1, c(3, 3), padding = "same") %>%
  layer_batch_normalization() %>%
  layer_activation_relu() %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  # Hidden layer 2
  layer_conv_2d(intermediate_dim2, c(3, 3), padding = "same") %>%
  layer_batch_normalization() %>%
  layer_activation_relu() %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  # Hidden layer 3
  layer_conv_2d(intermediate_dim3, c(3, 3), padding = "same", activation = "linear") %>%
  # Flatten the convolutional layer output to be fed to a dense layer
  layer_flatten()

z_mean    <- x %>% layer_dense(latent_dim, name = "z_mean")
z_log_var <- x %>% layer_dense(latent_dim, name = "z_log_var")

encoder <- keras_model(encoder_inputs, list(z_mean, z_log_var), name = "encoder")

encoder





# Sampler --------------------------------------------------
layer_sampler <- new_layer_class(
  classname = "Sampler",
  call = function(z_mean, z_log_var) {
    epsilon <- tf$random$normal(shape = tf$shape(z_mean))
    z_mean + exp(0.5 * z_log_var) * epsilon }
)





# Decoder ---------------------------------------------------

latent_inputs <- layer_input(shape = latent_dim)

decoder_outputs <- latent_inputs %>%
  # Reshape the input to add the dimensions needed for convolutional layers
  layer_reshape(target_shape = c(25, 2, 1)) %>%
  # Hidden layer 1
  layer_conv_2d_transpose(intermediate_dim3, c(3, 3), padding = "same", strides = 3) %>%
  layer_batch_normalization() %>%
  layer_activation_relu() %>%
  # Hidden layer 2
  layer_conv_2d_transpose(intermediate_dim2, c(3, 3), padding = "same") %>%
  layer_batch_normalization() %>%
  layer_activation_relu() %>%
  layer_upsampling_2d(size = c(2, 2)) %>%
  # Hidden layer 3
  layer_conv_2d_transpose(intermediate_dim1, c(3, 3), padding = "same", activation = "linear")

  


decoder <- keras_model(latent_inputs, decoder_outputs, name = "decoder")


decoder




# Loss (plus some extras)  ---------------------------------------

model_vae <- new_model_class(
  classname = "VAE",
  
  initialize = function(encoder, decoder, ...) {
    super$initialize(...)
    self$encoder <- encoder
    self$decoder <- decoder
    self$sampler <- layer_sampler()
    self$total_loss_tracker <-
      metric_mean(name = "total_loss")
    self$reconstruction_loss_tracker <-
      metric_mean(name = "reconstruction_loss")
    self$kl_loss_tracker <-
      metric_mean(name = "kl_loss")
  },
  
  metrics = mark_active(function() {
    list(
      self$total_loss_tracker,
      self$reconstruction_loss_tracker,
      self$kl_loss_tracker
    )
  }),
  
  train_step = function(data) {
    with(tf$GradientTape() %as% tape, {
      
      c(z_mean, z_log_var) %<-% self$encoder(data)
      z <- self$sampler(z_mean, z_log_var)
      
      reconstruction <- decoder(z)
      reconstruction_loss <-
        loss_mean_squared_error(data, reconstruction) %>%
        sum(axis = c(1)) %>%
        mean()
      
      kl_loss <- -0.5 * (1 + z_log_var - z_mean^2 - exp(z_log_var))
      total_loss <- reconstruction_loss + mean(kl_loss)
    })
    
    grads <- tape$gradient(total_loss, self$trainable_weights)
    self$optimizer$apply_gradients(zip_lists(grads, self$trainable_weights))
    
    self$total_loss_tracker$update_state(total_loss)
    self$reconstruction_loss_tracker$update_state(reconstruction_loss)
    self$kl_loss_tracker$update_state(kl_loss)
    
    list(total_loss = self$total_loss_tracker$result(),
         reconstruction_loss = self$reconstruction_loss_tracker$result(),
         kl_loss = self$kl_loss_tracker$result())
  }
)


# Train model ----------------------------------------------

# Create the VAE object
vae <- model_vae(encoder, decoder)
# Compile VAE object
vae %>% compile(optimizer_adam(learning_rate = 0.001))

# Train function
vae %>% fit(ds1.train.x, epochs = 50, batch_size = 128)
# Save the final weights of the trained model to a "weights" folder
vae %>% save_model_weights_tf("weights/12lead_2D_cnnvae")



# Test VAE / Plotting Beats  ------------------------------------------------------

# Create a VAE object
vae <- model_vae(encoder, decoder)
# Compile the object, Specifying LR does not matter since it is not going to be trained
vae %>% compile(optimizer_adam())
# Load the saved training weights to the VAE object;
load_model_weights_tf(vae, "D:/Documents/UCO/Research/Lead2 CNNVAE Weights/lead2_cnnvae")



# Use the loaded VAE to encode the current lead's data
x_test_encoded <- predict(vae$encoder, ds1.test.x, batch_size = 200)
eps <- rnorm(dim(ds1.test.x)[1])
# Sample from the current lead data's encoded representation
x_test_sampled <- x_test_encoded[[1]] + exp(0.5 * x_test_encoded[[2]]) * eps
# Decode the sampled vector; (Generate a beat reconstruction)
x_test_decoded <- predict(vae$decoder, x_test_sampled)


# Calculate the MSE of the current VAE
mse = mean((x_test_decoded - ds1.test.x)^2)



# Using lead 2 data
plot(1:150, x_test_decoded[12788,,2,1], type = "l", col = "red") # VAE beat in red
lines(1:150, ds1.test.x[12788,,2,1], col = "blue") # true beat in blue



