
setwd("C:/Users/Jyoti Prakash/Desktop")
task2data=read.csv("task2data.csv")
str(task2data)
data1=subset(task2data,select=-c(p_id))
library(rpart)
data1=na.omit(data1)
sum(is.na(data1))


for (i in names(data1)){
  if((class(data1[[i]])=="factor")|(class(data1[[i]])=="character")||(class(data1[[i]])=="logical")){
    data1[[i]]=as.numeric(factor(data1[[i]]))
  }
}
library(psych)
cor.ci(data1,method='spearman')

data2=subset(data1,select=-c(Q_pattern_recog,Q_attention_to_detail,Q_general_awareness,
                             Q_reading_comprehension,Q_logical_ability,P.score.other.data,
                             P.Score.ts,P.Score.Sd,P.Score.voice,P.Score.Qualifiers,
                             Q.score.other.data,Q.Score.ts,Q.Score.Sd,Q.Score.voice,
                             Q.Score.Qualifiers))

cor.ci(data2,method='spearman')

#Feature Selection using Boruta
library(Boruta)
set.seed(123)
data2=data.frame(data2)
boruta.train =Boruta(Average.Q.Score~., data = data2, doTrace = 2)
print(boruta.train)
final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)
impF=getSelectedAttributes(final.boruta, withTentative = F)
#Splitting data as train and test
library(caTools)
data2=subset(data2,select=-c(age,Gender,lives))
data2=data.frame(data2)
spl=sample.split(data2$Average.Q.Score,SplitRatio=0.7)
train=data2[spl==TRUE,]
test=data2[spl==FALSE,]
library(caret)
controlparameter=trainControl(method='cv',number=5)
train=data.frame(train)

model1=train(Average.Q.Score~.,data=data.frame(train),
             method='rpart',
             trControl=controlparameter
             )
summary(model1)
rpart.plot(model1$finalModel)
text(model1$finalModel)
pred1=predict(model1,newdata=test)
library(Metrics)
rmse(pred1,test$Average.Q.Score)


model2=train(Average.Q.Score~.,data=data.frame(train),
             method='rf',
             trControl=controlparameter
)
summary(model2)
pred2=predict(model2,newdata=test)
library(Metrics)
rmse(pred2,test$Average.Q.Score)

model3=train(Average.Q.Score~.,data=train,
             method='lm',
             trControl=controlparameter
)
summary(model3)
pred3=predict(model3,newdata=test)
rmse(pred3,test$Average.Q.Score)


#We have got a baseline and best model using RandomForest
#Using ensembling to improve the accuracy of the model 
#we will use 5 different models

# create submodels
library(caretEnsemble)
library(mlbench)
library(caret)
control <- trainControl(method="repeatedcv", number=10, 
                        savePredictions=TRUE, classProbs=FALSE)
algorithmList <- c('rf', 'rpart', 'xgbTree', 'glmnet')
set.seed(5)
models <- caretList(Average.Q.Score~., data=train, trControl=control, methodList=algorithmList)
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
rmse(predf,test$Average.Q.Score)
