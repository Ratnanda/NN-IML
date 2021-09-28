set.seed(100)
library(neuralnet) #membangun model neural network
library(caret) #confusion matrix
library(MLmetrics) #evaluasi model
library(iml) #interpretable machine learning
library(ggplot2) #memunculkan plot

setwd("D:/TUGAS AKHIR")
datatrain = read.csv("train_size0.csv")
datatest = read.csv("test_size0.csv")
str(datatrain) #melihat tipe data training
str(datatest) #melihat tipe data testing

#Menentukan Dataset Prediktor dan Target
X.train <- datatrain[,-19]
Y.train <- datatrain[,19]
X.test <- datatest[,-19]
Y.test <- datatest[,19]

#Transformasi data
X_train <- scale(X.train)
X_test <- scale(X.test)
str(X_test)

#Mengubah tipe data variabel target menjadi target
FD = as.factor(datatrain$FD)
FD_test = as.factor(datatest$FD)
train = data.frame(X_train,FD)
test = data.frame(X_test,FD_test)
str(train)
str(test)

#Training neural network classification
set.seed(100)
nn_model = neuralnet(FD ~ . , data = train, 
                     hidden = c(10), #banyak neuron setiap layer
                     linear.output = FALSE #output klasifikasi
                     )

#Predicting testing
net.predict = predict(nn_model, X_test) #probabilitas kelas prediksi
net.prediction = c("0", "1")[apply(net.predict, 1, which.max)] #klasifikasi kelas prediksi
predict.table = table(test$FD_test, net.prediction) #tabel confusion matrix
predict.table

#Evaluation testing
confusionMatrix(predict.table)
F1_Score(test$FD_test,net.prediction)
Accuracy(net.prediction,test$FD_test)
AUC(test$FD_test,net.prediction)

#Predicting training
nn.predict.train = predict(nn_model, X_train)
nn.prediction.train = c("0", "1")[apply(nn.predict.train, 1, which.max)]
predict.table.train = table(train$FD, nn.prediction.train)
predict.table.train

#Evaluation training
confusionMatrix(predict.table.train)
F1_Score(train$FD,nn.prediction.train)
Accuracy(nn.prediction.train,train$FD)
AUC(train$FD,nn.prediction.train)

##########################################################################

#Membangun fungsi prediksi
pred.fun.nn = function(obj, newdata) {
  pred = predict(obj, newdata)
  result = data.frame(pred)
  colnames(result)=c("Aman", "Financial_Distress")
  return(result)
}
prediksi = pred.fun.nn(nn_model,train)
head(prediksi)


#Predictor
set.seed(100)
mod.nn.fd = Predictor$new(nn_model, 
                        data = train, y ="FD", 
                        predict.function = pred.fun.nn,
                        class = "Financial_Distress")
#PDP
pdp.nn <- FeatureEffects$new(mod.nn.fd, method = "pdp")
pdp.nn$plot() + labs(caption = "PDP NN Size 0")

#Feature Interaction
set.seed(100)
interact.nn = Interaction$new(mod.nn.fd)
plot(interact.nn) +  ggtitle("NN")  + labs(caption = "Size 0") + theme_bw()

#Feature Important
set.seed(100)
imp.nn <- FeatureImp$new(mod.nn.fd, loss = "f1", compare = "difference")
plot(imp.nn) + ggtitle("Feature Important NN Size 0")

#TreeSurrogate
mod.nn = Predictor$new(nn_model , 
                       data = train, y ="FD", 
                       predict.function = pred.fun.nn)
dt.nn <- TreeSurrogate$new(mod.nn, maxdepth = 4)
dt.nn$r.squared
plot(dt.nn) + labs(caption = "Size 0") + theme_bw() +ggtitle("NN")
plot(dt.nn$tree)

#LOCAL INTERPRETATION
#shapley
head(train)
x.int1.nn <- train[1, ] #perusahaan amatan lokal FD = 1
x.int0.nn <- train[2, ] #perusahaan amatan lokal FD = 0

set.seed(100)
shap1.nn <- Shapley$new(mod.nn.fd, x.interest = x.int1.nn)
shap1.nn$results
plot(shap1.nn) + labs(caption = "NN Size 0 {FD=1}") + theme_light()

set.seed(100)
shap0.nn <- Shapley$new(mod.nn.fd, x.interest = x.int0.nn)
shap0.nn$results
plot(shap0.nn) + labs(caption = "NN Size 0 {FD=0}") + theme_light()


#LIME
lime1.nn <- LocalModel$new(mod.nn.fd, x.interest = x.int1.nn, k = 17)
lime1.nn$results
plot(lime1.nn) + labs(caption = "NN Size 0 {FD=1}") 

lime0.nn <- LocalModel$new(mod.nn.fd, x.interest = x.int0.nn, k = 17)
lime0.nn$results
plot(lime0.nn) + labs(caption = "NN Size 0 {FD=0}") 


