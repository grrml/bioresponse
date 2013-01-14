function [prob prob1] = test_example_NN(train_x,train_y,test_x,test_y,test_xr,test_yr)
addpath(genpath('../.'))
%%load mnist_uint8;

%%train_x = double(train_x) / 255;
%%test_x  = double(test_x)  / 255;
%%train_y = double(train_y);
%%test_y  = double(test_y);
%% ex1 vanilla neural net
%rng(0);
%randperm(0);

%nn = nnsetup([784 100 10]);

%nn.learningRate = 1;   %  Learning rate
%opts.numepochs =  50;   %  Number of full sweeps through data
%opts.batchsize = 100;  %  Take a mean gradient step over this many samples
%opts.silent = 1;
%nn = nntrain(nn, train_x, train_y, opts);

%[er, bad, prob] = nntest(nn, test_x, test_y);
%er
%%[er, bad, prob1] = nntest(nn, test_xr, test_yr);
%er
%assert(er < 0.1, 'Too big error');

%% ex2 neural net with L2 weight decay
%rng(0);
randperm(0);
if(size(train_x,1)==3375)
    tmpnb = 15;
else
    tmpnb = 16;
end

nn = nnsetup([1776 100 2]);

nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
nn.learningRate = 1;        %  Learning rate
opts.numepochs =  50;        %  Number of full sweeps through data
opts.batchsize = tmpnb;       %  Take a mean gradient step over this many samples
opts.silent = 1;
nn = nntrain(nn, train_x, train_y, opts);

[er, bad, prob] = nntest(nn, test_x, test_y);
[er, bad, prob1] = nntest(nn, test_xr, test_yr);
%er
%assert(er < 0.1, 'Too big error');

%% ex3 neural net with dropout
%for i=1:20
%%randperm(0);

%if(size(train_x,1)==3375)
%    tmpnb = 15;
%else
%    tmpnb = 16;
%end

%%nn = nnsetup([784 100 10]);

%%nn.dropoutFraction = 0.5;   %  Dropout fraction 
%%nn.learningRate = 1;        %  Learning rate
%%opts.numepochs =  50;        %  Number of full sweeps through data
%%opts.batchsize = 100;       %  Take a mean gradient step over this many samples
%%opts.silent = 1;
%%nn = nntrain(nn, train_x, train_y, opts);

%%[er, bad, prob] = nntest(nn, test_x, test_y);
%%er
%[er, bad, prob1] = nntest(nn, test_xr, test_yr);
%disp(['Err-NN',num2str(er)])
%end
%assert(er < 0.16, 'Too big error');
