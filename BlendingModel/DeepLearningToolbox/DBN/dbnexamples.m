clear all; close all; clc;

load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

Train = importdata('train.csv');
Train = Train.data(1:end-1,:);
TrainLabel = Train(:,1);
Train = Train(:,2:end);
Test = importdata('test.csv');
Test = Test.data;
TestLabel = importdata('TestLabel.csv');
%%  ex1 train a 100 hidden unit RBM and visualize its weights
dbn.sizes = [100];
opts.numepochs =   5;
opts.batchsize = 50;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, Train, opts);
dbn = dbntrain(dbn, Train, opts);
figure; visualize(dbn.rbm{1}.W', 1);   %  Visualize the RMB weights

%%  ex2 train a 100-100-100 DBN and use its weights to initialize a NN
dbn.sizes = [100 100 100];
opts.numepochs =   100;
opts.batchsize = 50;
opts.momentum  =   0;
opts.alpha     =   1;
size(Train)
dbn = dbnsetup(dbn, Train, opts);
dbn = dbntrain(dbn, Train, opts);

nn = dbnunfoldtonn(dbn, 1);

nn.alpha  = 1;
nn.lambda = 1e-4;
opts.numepochs =  100;
opts.batchsize = 50;

%size(train_x)
%size(train_y)
nn = nntrain(nn, Train, TrainLabel, opts);
[er, bad] = nntest(nn, Test, TestLabel);

disp([num2str(er * 100) '% error']);
figure; visualize(nn.W{1}', 1);
