addpath(genpath('../.'))
addpath data;
%%
%Pre-Processing
train=importdata('train.csv');
train_y=train.data(:,1);
train_x=train.data(:,2:end);
test=importdata('test.csv');
test_x = test.data;
test_y=importdata('Testlabel.csv');
tempy=zeros(size(test_x,1),1);
for i=1:size(test_x,1)
    if(test_y(i,1)==1)
        tempy(i,1)=0;
    else
        tempy(i,1)=1;
    end
end
test_y=[test_y,tempy];

blendTrain = zeros(size(train_x,1),3);
blendTest = zeros(size(test_x,1),3);

nFold = 10;

X_train = zeros(size(train_x,1),1776);
X_test = zeros(size(test_x,1),1776);
y_train = zeros(size(train_y,1),2);

for j=1:3
    disp(['Model-',num2str(j)])
    blendTestj = zeros(size(test_x,1),nFold);
    for i=1:nFold
        disp(['Validation Fold-',num2str(i)])
        fnametrain = strcat('idtrain1749',num2str(i-1));
        fnametrain = strcat(fnametrain,'.csv');
        fnametest = strcat('idtest1749',num2str(i-1)); 
        fnametest = strcat(fnametest,'.csv');
        idtrain = importdata(fnametrain);
        idtest = importdata(fnametest);
        
        X_train = train_x(idtrain+1,:);
        y_train = train_y(idtrain+1,:);
        X_test = train_x(idtest+1,:);
        y_test = train_y(idtest+1,:);
        
        tempy=zeros(size(y_train,1),1);
        for k=1:size(y_train,1)
            if(y_train(k,1)==1)
                tempy(k,1)=0;
            else
                tempy(k,1)=1;
            end
        end
        y_train=[y_train,tempy];
        
        tempy=zeros(size(y_test,1),1);
        for k=1:size(y_test,1)
            if(y_test(k,1)==1)
                tempy(k,1)=0;
            else
                tempy(k,1)=1;
            end
        end
        y_test=[y_test,tempy];
        
        if(j==1)
            [y_submission y_s] = test_example_DBN(X_train,y_train,X_test,y_test, test_x, test_y);
        elseif(j==2)
            [y_submission y_s] = test_example_SAE(X_train,y_train,X_test,y_test, test_x, test_y);
        else
            [y_submission y_s] = test_example_NN(X_train,y_train,X_test,y_test, test_x, test_y);    
        end
        
        blendTrain(idtest+1,j) = y_submission;
        blendTestj(:,i) = y_s;
    end
    blendTest(:,j) = mean(blendTestj,2);
end

size(blendTrain)
size(blendTest)
xlswrite('blendtrain1749v2.csv',blendTrain);
xlswrite('blendtest1749v2.csv',blendTest);


