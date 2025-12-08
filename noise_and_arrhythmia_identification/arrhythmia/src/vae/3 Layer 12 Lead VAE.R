############################ 12-Lead Vanilla VAE ###############################
library(keras)

load("D:/Documents/UCO/Research/Data Sets/INCART12/ord_data.RData")
ord.ds1.x.train = unname(ord.ds1.x.train)
ord.ds1.x.test = unname(ord.ds1.x.test)

# Variables for all 12 VAEs
input_size = 150
latent_space_dim = 50

############################# 3 Layer Vanilla VAE ##############################
# ENCODER
encoder_input = layer_input(shape = input_size, name = "encoder_input")
encoder_ff_layer1 = layer_dense(units = 150, activation = "relu", name = "encoder_ff_1")(encoder_input)
# Hidden Layer 1
encoder_ff_layer2 = layer_dense(units = 125, activation = "relu", name = "encoder_ff_2")(encoder_ff_layer1)
# Hidden Layer 2
encoder_ff_layer3 = layer_dense(units = 100, activation = "relu", name = "encoder_ff_3")(encoder_ff_layer2)
# Hidden Layer 3
encoder_ff_layer4 = layer_dense(units = 75, activation = "relu", name = "encoder_ff_4")(encoder_ff_layer3)
# creating output layers whose vector outputs are treated as mean and variance
encoder_mu = layer_dense(units = latent_space_dim, name = "encoder_mu")(encoder_ff_layer4)
encoder_log_variance = layer_dense(units =latent_space_dim, name = "encoder_log_variance")(encoder_ff_layer4)
# Encoder model outputs the mu and variance layer outputs
encoder_3layer = keras_model(encoder_input, list(encoder_mu, encoder_log_variance), name = "encoder_model")
summary(encoder_3layer)

# DECODER
decoder_input = layer_input(shape = latent_space_dim, name = "decoder_input")
decoder_ff_layer1 = layer_dense(units = latent_space_dim, activation = "relu", name = "decoder_ff_1")(decoder_input)
# Hidden Layer 1
decoder_ff_layer2 = layer_dense(units = 75, activation = "relu", name = "decoder_ff_2")(decoder_ff_layer1)
# Hidden Layer 1
decoder_ff_layer3 = layer_dense(units = 100, activation = "relu", name = "decoder_ff_3")(decoder_ff_layer2)
# Hidden Layer 1
decoder_ff_layer4 = layer_dense(units = 125, activation = "relu", name = "decoder_ff_4")(decoder_ff_layer3)
# Decoder output layer
decoder_ff_layer5 = layer_dense(units = 150, activation = "linear", name = "decoder_ff_5")(decoder_ff_layer4)
# Compile the decoder into a keras model
decoder_3layer = keras_model(decoder_input, decoder_ff_layer5, name = "decoder_model")
summary(decoder_3layer)

######################################## SAMPLE LAYER ##########################
# creating the sampling function for encoder output
Sampling = function(z_mean, z_log_var) {
  batch = k_shape(z_mean)[1]
  dim = k_shape(z_mean)[2]
  epsilon = k_random_normal(shape = c(batch, dim))
  return(z_mean + k_exp(0.5 * z_log_var) * epsilon)
}
# Mu Input Layer
mu_input = layer_input(shape = latent_space_dim, name = "mu_input")
# Variance Input Layer
var_input = layer_input(shape = latent_space_dim, name = "var_input")
sampling_layer = layer_lambda(f = function(x) Sampling(x[[1]], x[[2]]), name = "sample_output")(list(mu_input, var_input))
# Create Sampler keras model
sampler = keras_model(list(mu_input, var_input), sampling_layer, name = "sample_model")
summary(sampler)

##################### COMBINING LAYERS INTO VAE_3Layer MODEL ########################
# VAE input layer
vae_3layer_input = layer_input(shape = input_size, name="VAE_input")
# Connecting encoder to the input layer
vae_3layer_encoder = encoder_3layer(vae_3layer_input)
# Connecting sample layer to encoder
vae_3layer_sampler = sampler(vae_3layer_encoder)
# Connecting decoder to the encoder output
vae_3layer_decoder = decoder_3layer(vae_3layer_sampler)
# Instantiating final model
vae_3layer = keras_model(vae_3layer_input, vae_3layer_decoder, name = "VANILLAVAE1")
summary(vae_3layer)

################################# LOSS FUNCTION ###############################
loss_func = function(encoder_mu, encoder_log_variance) {
  #Returns Reconstruction loss using MSE
  vae_reconstruction_loss = function(y_true, y_predict) {
    reconstruction_loss = k_mean(k_square(y_true - y_predict), axis = 2)
    return(reconstruction_loss)
  }
  # Returns KL loss
  vae_kl_loss = function(encoder_mu, encoder_log_variance) {
    kl_loss = -0.5 * k_sum(1.0 + encoder_log_variance - k_square(encoder_mu) - k_exp(encoder_log_variance), axis = 2)
    return(kl_loss)
  }
  # Returns total model loss, which is Reconstruction + KL loss
  vae_loss = function(y_true, y_predict) {
    reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
    kl_loss = vae_kl_loss(y_true, y_predict)
    loss = reconstruction_loss + kl_loss
    return(loss)
  }
  return(vae_loss)
}

################################# Model Compilation ############################
# Compile 3 layer VAE
vae_3layer %>% compile(loss = loss_func(encoder_mu, encoder_log_variance), 
                       optimizer = optimizer_adam(learning_rate = 0.01))

################################## MODEL TRAINING ##############################
# Stops model training early if the loss doesnt change by 0.01 after 5 epochs
early_stopping = callback_early_stopping(
  monitor = "loss",
  min_delta = 0.01,
  patience = 5,
  verbose = 1
)

# Train a 3 layer vae for each lead
results = list()
for (i in 1:12){
  history = vae_3layer %>% fit(ord.ds1.x.train[,,i], ord.ds1.x.train[,,i], 
                               epochs = 50, batch_size = 128, callbacks = list(early_stopping))
  vae_3layer_predict = vae_3layer %>% predict(ord.ds1.x.test[,,i])
  test_loss = mean((ord.ds1.x.test[,,i] - vae_3layer_predict) ** 2)
  results[[i]] = list(leads = i, test_loss = test_loss)
  print(mean(test_loss))
}

#print(ord.ds1.x.test[,,i])

# Save training results to csv file
results = data.frame(results)
write.csv(results, file = "/home/efogarty/Documents/Grid Search Results/3 Layer 12 Lead VAE Results.csv",
          row.names = FALSE)


