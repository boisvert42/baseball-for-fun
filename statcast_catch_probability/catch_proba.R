# Download data from pages like https://goo.gl/hYRsN9
# Put them in a folder called statcast_data
filenames <- list.files(path = "statcast_data", full.names=TRUE)
# Combine them into a single data.frame
data_raw <- do.call("rbind", lapply(filenames, read.csv, header = TRUE))

# Exclude home runs
data2 <- data_raw[data_raw$events != 'Home Run',]
# Take only balls that make it to the outfield
data <- data2[data2$hit_location %in% c(7,8,9),c("game_date","events","hit_speed","hit_angle","hit_location","des","hc_x","hc_y")]

# Look at the data
str(data)

# We need to massage the data before we can use it
# Note that "hit_speed" and "hit_angle" are factors; we have to change that
data$hit_speed <- as.numeric(as.character(data$hit_speed))
data$hit_angle <- as.numeric(as.character(data$hit_angle))

# Add in horizontal angle (thanks to Bill Petti)
# We're not actually going to do this because it turns out it doesn't help
horiz_angle <- function(df) {
    angle <- with(df, round(tan((hc_x-128)/(208-hc_y))*180/pi*.75,1))
    angle
}
#data$hor_angle <- horiz_angle(data)

# We got some "NA"s doing the above.  Let's get rid of them.
data <- na.omit(data)

# Look at the data
str(data)
# These are the "outs"
# Note that using "In play, out(s)" doesn't work!  
# It includes stuff like assists at third base on a single
data$out <- as.factor(data$events %in% c('Flyout','Lineout','Sac Fly','Sac Fly DP'))

goodColumns <- c('hit_speed','hit_angle','out')
library(caret)
# Set up training and test sets
inTrain <- createDataPartition(data$out,p=0.8,list=FALSE)
training <- data[inTrain,goodColumns]
testing <- data[-inTrain,goodColumns]

method <- 'glm'; N <- 20 # logistic regression -- pair with "binomial" as below
ctrl <- trainControl(method = 'repeatedcv',number = N, repeats = N)
modelFit <- train(out ~ ., method=method, data=training, family='binomial',trControl=ctrl)

# How did this work on the test set?
predicted <- predict(modelFit,newdata=testing)
# Accuracy, precision, recall, F1 score
accuracy <- sum(predicted == testing$out)/length(predicted)
precision <- posPredValue(predicted,testing$out)
recall <- sensitivity(predicted,testing$out)
F1 <- (2 * precision * recall)/(precision + recall)

print(accuracy) # 0.8385
print(precision) # 0.8338
print(recall) # 0.8671
print(F1) # 0.8501

# Print the coefficients so we can do this in Excel
print(modelFit$finalModel$coefficients)
