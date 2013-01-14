function [prob prob1] = test_example_DBN(train_x,train_y,test_x,test_y,test_xr, test_yr)
%load mnist_uint8;
addpath(genpath('../.'))

%%  ex1 train a 100 hidden unit RBM and visualize its weights
randperm(0);
if(size(train_x,1)==3375)
    tmpnb = 15;
else
    tmpnb = 16;
end
dbn.sizes = [100];
opts.numepochs =   50;
opts.batchsize = tmpnb;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
nn = dbnunfoldtonn(dbn, 2);

%train nn
nn.learningRate  = 1;
opts.numepochs =  50;
opts.batchsize = tmpnb;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad, prob] = nntest(nn, test_x, test_y);
[er, bad, prob1] = nntest(nn, test_xr, test_yr);
%figure; visualize(dbn.rbm{1}.W', 1);   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
%randperm(0);
%train dbn
%if(size(train_x,1)==3375)
%    tmpnb = 15;
%else
%    tmpnb = 16;
%end

%dbn.sizes = [200 200];
%opts.numepochs =   50;
%opts.batchsize = tmpnb;
%opts.momentum  =   0;
%opts.alpha     =   1;
%dbn = dbnsetup(dbn, train_x, opts);
%dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
%nn = dbnunfoldtonn(dbn, 2);

%train nn
%nn.learningRate  = 1;
%opts.numepochs =  50;
%opts.batchsize = tmpnb;
%nn = nntrain(nn, train_x, train_y, opts);
%[er, bad, prob] = nntest(nn, test_x, test_y);
%[er, bad, prob1] = nntest(nn, test_xr, test_yr);
%disp(['Err-DBN',num2str(er)])
%assert(er < 0.12, 'Too big error');