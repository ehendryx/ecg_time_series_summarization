# Date:08/21/2024
# Name: Ethan F.
# Description: RNNVAE for arrhythmia data


library(tensorflow)
library(keras)
library(reticulate)
library(dplyr) # Needed for %>% operator


# DS1 DATA LOADING -------------------------------------------------------------
load("/home/efogarty/Documents/Data Sets/INCART 12/Fully Processed INCART Data - Bundle and Escape as Arrhythmia.RData")

ds1_train_data = ds1.train.x[,,2]
# Reshaping data to match model input layer
ds1_train_data = array_reshape(ds1_train_data, c(dim(ds1_train_data), 1))



# Variables --------------------------------------------------------------------
input_size = 150
channels = 1
original_dim = c(input_size, channels)
latent_dim = 100
tensorflow::set_random_seed(3)



#  Inputs and encoder -----------------------------------------------------

encoder_inputs <- layer_input(shape = c(original_dim))

x <- encoder_inputs %>%
  layer_lstm(units = 1, activation = "tanh", return_sequences = TRUE) %>%
  layer_lstm(units = 5, activation = "tanh", return_sequences = TRUE) %>%
  layer_lstm(units = 10, activation = "tanh", return_sequences = TRUE) %>%
  # Flatten the convolutional layer output to be fed to a dense layer
  layer_flatten()

z_mean    <- x %>% layer_dense(latent_dim, name = "z_mean")
z_log_var <- x %>% layer_dense(latent_dim, name = "z_log_var")
encoder <- keras_model(encoder_inputs, list(z_mean, z_log_var), name = "encoder")

summary(encoder)





# Sampler --------------------------------------------------
layer_sampler <- new_layer_class(
  classname = "Sampler",
  call = function(z_mean, z_log_var) {
    epsilon <- tf$random$normal(shape = tf$shape(z_mean))
    z_mean + exp(0.5 * z_log_var) * epsilon }
)





# Decoder ---------------------------------------------------

latent_inputs <- layer_input(shape = c(latent_dim))

decoder_outputs <- latent_inputs %>%
  layer_dense(units = prod(150, 10)) %>%
  # Reshape the input to add the 3rd dimension needed for convolutional layers
  layer_reshape(target_shape = c(150, 10)) %>%
  layer_lstm(units = 10, activation = "tanh", return_sequences = TRUE) %>%
  layer_lstm(units = 5, activation = "tanh", return_sequences = TRUE) %>%
  layer_lstm(units = 1, activation = "tanh", return_sequences = TRUE)


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



# Train model -----------------------------------------------------------------

# Create the VAE object
vae <- model_vae(encoder, decoder)
# Compile VAE object
vae %>% compile(optimizer_adam(learning_rate = 0.001))


# Train function
vae %>% fit(ds1_train_data, epochs = 50, batch_size = 128)
# Save the final weights of the trained model to a "weights" folder
vae %>% save_model_weights_tf("/home/efogarty/Documents/R Scripts/Summer2024/Arrhythmia Detection/Arrhythmia RNNVAE 2 Weights/Arrhythmia RNNVAE")




# Predict with Model -----------------------------------------------------------

ds1_test_data = ds1.test.x[,,2]
# Reshaping data to match model input layer
ds1_test_data = array_reshape(ds1_test_data, c(dim(ds1_test_data), 1))


# Use the loaded VAE to encode the current lead's data
x_test_encoded <- predict(vae$encoder, ds1_test_data, batch_size = 200)
eps <- rnorm(dim(ds1_test_data)[1])
# Sample from the current lead data's encoded representation
x_test_sampled <- x_test_encoded[[1]] + exp(0.5 * x_test_encoded[[2]]) * eps
# Decode the sampled vector; (Generate a beat reconstruction)
x_test_decoded <- predict(vae$decoder, x_test_sampled)


# Calculate the MSE of the current VAE
mse = mean((x_test_decoded - ds1_test_data)^2)


# Using lead 2 data
plot(1:150, ds1_test_data[600,,], type = "l", col = "red", xlab = "Sample Number", ylab = "mV") # True beat in red
lines(1:150, x_test_decoded[600,,], col = "blue") # VAE beat in blue


rm(list=ls())
gc(reset=TRUE)
gc()
