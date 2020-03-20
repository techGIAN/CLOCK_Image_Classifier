# CLOCK Model
# CLassifier for Outdoor images using Cnns through Keras
# Author: Gian Carlo Alix
# Visit my GitHub Repo of CLOCK: https://github.com/techGIAN/CLOCK_Image_Classifier

# necessary imports for image processing
library(plyr)
library(keras)
library(EBImage)
library(stringr)
library(pbapply)
library(tensorflow)

# Want to measure how long preprocessing is
t1 <- Sys.time()

# load the photoMetaData
meta_data <- read.csv("../photoMetaData.csv")

# query on the first few rows of the meta dataset
head(meta_data)

# query the number of images we have
n_images <- nrow(meta_data)
n_images

# re-assign categorical labels to numerical labels in the meta_data
unique_labels <- c("non-outdoor", "outdoor")
unique_labels_count <- length(unique_labels)
for (i in 1:n_images) {
  # this is for Problem II
  # if the category of the image has "outdoor" in its name, then assign label of 1, and 0 otherwise
  meta_data$label[i] <- if(grepl("outdoor",meta_data$category[i],fixed=TRUE)) 1 else 0
}

count(meta_data$label)

# view the first image in our image dataset
sample_IMG <- readImage("../columbiaImages/CRW_4786_JFR.jpg")
sample_IMG
display(sample_IMG)
meta_data[1,]

# new dimensions of image when scaled
new_width <- 32
new_height <- 32
channels <- 3   # represents the RGB Channel

# read all images and store in a 4d tensor
img_set <- array(NA, dim=c(n_images,new_height,new_width,channels))
for (i in 1:n_images) {
  filename <- paste0("../columbiaImages/",meta_data$name[i])
  image <- readImage(filename)
  img_resized <- resize(image, w = new_width, h = new_height)
  img_set[i,,,] = imageData(img_resized)
}

# split the image and meta dataset into training and testing
# set.seed(100)
train_split <- 0.8
validation_split <- 0.5
train_sample_size <- floor(train_split*n_images)
arr <- 1:n_images
train = sample(arr,train_sample_size)
train_set_x = img_set[train,,,]

not_training <- arr[!arr %in% train]
validation_sample_size <- floor(validation_split*length(not_training))
validation = sample(not_training,validation_sample_size)
validation_set_x = img_set[validation,,,]

not_tr_te <- not_training[!not_training %in% validation]
test_set_x = img_set[not_tr_te,,,]

meta_train = meta_data[train,]
meta_validation = meta_data[validation,]
meta_test = meta_data[not_tr_te,]

train_set_y_cat = as.matrix(meta_train[ncol(meta_data)])
validation_set_y_cat = as.matrix(meta_validation[ncol(meta_data)])
test_set_y_cat = as.matrix(meta_test[ncol(meta_data)])

# use keras' built-in function for one-hot encoding (mostly for Problem III but can also be used for Problem II)
train_set_y <- to_categorical(train_set_y_cat,num_classes=unique_labels_count)
validation_set_y <- to_categorical(validation_set_y_cat,num_classes=unique_labels_count)
test_set_y <- to_categorical(test_set_y_cat,num_classes=unique_labels_count)

# ==================================================================================================================================
# ==================================================================================================================================
# TRAIN HERE

t2 <- Sys.time()

# then we build the architecture of our convolutional neural network
# begin by defining a sequential linear stack of layers
cnn_model <- keras_model_sequential()

# set the hyperparameters
# this can be tweaked to improve accuracy of the model
filter_size <- 100
kernel_width <- 3
kernel_height <- 3
kernel_dim <- c(kernel_width, kernel_height)
input_shape <- c(new_width, new_height, 3)
activation_func_1 <- "relu"
pool_width <- 2
pool_height <- 2
pool_dim <- c(pool_width, pool_height)
s_strides <- 3
dropout_val1 <- 0.8
dropout_val2 <- 0.5 
hidden_units_1 <- 1000
output_units <- unique_labels_count
activation_func_2 <- "softmax"
learning_rate <- 0.0001
learning_decay <- 10^-6

cnn_model %>%
  # 1st convolutional layer
  layer_conv_2d(filter=filter_size, kernel_size=kernel_dim, padding="same", input_shape=input_shape) %>%
  layer_activation(activation_func_1) %>%
  
  # Use a max pool layer to dimensionally reduce the feature map for the model's complexity reduction
  layer_max_pooling_2d(pool_size=pool_dim) %>%
  
  # Using a dropout to avoid overfitting
  layer_dropout(dropout_val1) %>%
  
  # need to flatten input
  layer_flatten() %>%
  
  # create a densely-connected neural network with 1000 hidden units with the given activation function relu (hidden layer)
  # and a 0.5 dropout, then for the output layer use softmax (to calculate cross-entropy)
  layer_dense(hidden_units_1) %>%
  layer_activation(activation_func_1) %>%
  layer_dropout(dropout_val2) %>%
  
  layer_dense(output_units) %>%
  layer_activation(activation_func_2)

optimizer <- optimizer_adamax(lr=learning_rate,decay=learning_decay)

# binary_crossentropy works really well for Problem II than logistic_crossentropy
# use logistic_crossentropy for Problem III
cnn_model %>%
  compile(loss="binary_crossentropy", optimizer=optimizer, metrics="accuracy")

# a summary of the architecture
summary(cnn_model)

# We then train our model
data_augment <- TRUE
batch_size <- 100
n_epochs <- 100
if (!data_augment) {
  cnn_model %>% fit(train_set_x, train_set_y, batch_size=batch_size, epochs=n_epochs,
                    validation_data=list(validation_set_x, validation_set_y), shuffle=TRUE)
} else {
  # Generate Images
  gen <- image_data_generator(featurewise_center=TRUE, featurewise_std_normalization=TRUE)
  
  # Fit the image data generator to the training data
  gen %>% fit_image_data_generator(train_set_x)
  
  # Generate batches of augmented/normalized data from image data and labels to visualize the
  # generated images made by the CNN Model
  cnn_model %>% fit_generator(flow_images_from_data(train_set_x, train_set_y, gen, batch=batch_size), steps_per_epoch=as.integer(train_sample_size/batch_size), epochs=n_epochs, validation_data=list(validation_set_x,validation_set_y))
  
}

t3 <- Sys.time()

# Check accuracy
model_score <- cnn_model %>% evaluate(test_set_x, test_set_y, verbose=0)
cat("Test Loss: ", model_score$loss, "\n")
cat("Test Accuracy: ", model_score$acc, "\n")

# View one test img (simple random sample)
r <- sample(1:length(not_tr_te),1)

img_name <- as.character(meta_test[r,]$name)
filename <- paste("../columbiaImages/", img_name, sep="")
display(readImage(filename))

category_predictions <- cnn_model %>% predict_classes(test_set_x)
p <- unique_labels[category_predictions[r]+1]
a <- unique_labels[as.numeric(test_set_y_cat[r])+1]

# determines whether the model was able to predict the class of the image correctly 
cat(img_name, "\n")
cat("Predicted:", p, "\n")
cat("Actual:", a, "\n")
match <- p == a
cat("The model predicted it correctly:", match, "\n")

t4 <- Sys.time()
preprocessing_time <- as.numeric(t2-t1)
training_time <- as.numeric(t3-t2)
accuracy_time <- as.numeric(t4-t3)

# displays run times for statistic purposes
cat("Data Preprocessing Time:", preprocessing_time, "secs\n")
cat("Model Training Time:", training_time, "secs\n")
cat("Calculating Accuracy Time:", accuracy_time, "secs\n")
