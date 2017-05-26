setwd("C:/Users/Jyoti Prakash/Desktop")
task1data=read.csv("taskdata.csv",na.strings="")
str(task1data)
data1=subset(task1data,select=-c(Id,city,is_banned))
library(rpart)
data1=na.omit(data1)
sum(is.na(data1))


hist(data1$No..of.Tasks)
#Plot is skewed, taking log transforms helps in making the plot normal
data1$logNo.of.task=log(data1$No..of.Tasks)

#Similarly for other parameters
data1$logEarning=log(data1$Earning)
data1$logEarningtilldate=log(data1$Earnings.till.date)

boxplot(logEarning~Ref.Source,data1)
boxplot(logEarning~State,data1)
boxplot(logEarning~gender,data1)

#one hot encoding for state
library(caret)
endata=subset(data1,select=c(state,gender,Ref.Source))
dummies = dummyVars(~., data =endata)
data.code1=as.data.frame(predict(dummies,newdata=endata))
data1=cbind(data1,data.code1)
data1=subset(data1,select=-c(state,gender,Ref.Source,Earning,Earnings.till.date))

#removing infinity values
is.na(data1) <- sapply(data1,is.infinite)
data1=na.omit(data1)



library(psych)
cor.ci(data1,method='spearman')

data2=data1

cor.ci(data2,method='spearman')

#Feature Selection using Boruta
library(Boruta)
set.seed(123)
data2=data.frame(data2)
boruta.train =Boruta(logEarning~., data = data2, doTrace = 2)
print(boruta.train)
final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)
impF=getSelectedAttributes(final.boruta, withTentative = F)
#Splitting data as train and test
library(caTools)
data2=subset(data2,select=c(impF,"logEarning"))
spl=sample.split(data2$logEarning,SplitRatio=0.7)
train=data2[spl==TRUE,]
test=data2[spl==FALSE,]
library(caret)
controlparameter=trainControl(method='cv',number=5)
train=data.frame(train)

model1=train(logEarning~.,train,
             method='rpart',
             trControl=controlparameter
)
summary(model1)
library(rpart.plot)
rpart.plot(model1$finalModel)
pred1=predict(model1,newdata=test)
library(Metrics)
rmse(pred1,test$logEarning)


model2=train(logEarning~.,data=train,
             method='rf',
             trControl=controlparameter
)
summary(model2)
pred2=predict(model2,newdata=test)
rmse(pred2,test$logEarning)

model3=train(logEarning~.,data=train,
             method='lm',
             trControl=controlparameter
)
summary(model3)
pred3=predict(model3,newdata=test)
rmse(pred3,test$logEarning)


#We have got a baseline and best model using RandomForest
#Using ensembling to improve the accuracy of the model 
#we will use 5 different models

# create submodels
library(caretEnsemble)
library(mlbench)
library(caret)
control <- trainControl(method="repeatedcv", number=10, 
                        savePredictions=TRUE, classProbs=FALSE)
algorithmList <- c('rf', 'rpart', 'lm', 'glmnet')
set.seed(5)
models <- caretList(logEarning~., data=train, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)


#model corelations
modelCor(results)
splom(results)


# stack using randomforest
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=FALSE)
set.seed(20)
stack.rf <- caretStack(models, method="rf", metric="RMSE", trControl=stackControl)
print(stack.rf)

predf=predict(stack.rf,newdata=test)
rmse(predf,test$logEarning)
