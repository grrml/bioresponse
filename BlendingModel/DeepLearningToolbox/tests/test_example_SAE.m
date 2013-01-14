function [prob prob1] = test_example_SAE(train_x,train_y,test_x,test_y, test_xr, test_yr)
addpath(genpath('../.'))

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
%rng(0);
randperm(0);

if(size(train_x,1)==3375)
    tmpnb = 15;
else
    tmpnb = 16;
end


sae = saesetup([1776 100]);
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   50;
opts.batchsize = tmpnb;%100;
sae = saetrain(sae, train_x, opts);
%visualize(sae.ae{1}.W{1}', 1)

% Use the SDAE to initialize a FFNN
nn = nnsetup([1776 100 2]);
nn.W{1} = sae.ae{1}.W{1};
nn.b{1} = sae.ae{1}.b{1};

% Train the FFNN 
nn.learningRate  = 1;
opts.numepochs =   50;
opts.batchsize = tmpnb;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad, prob] = nntest(nn, test_x, test_y);
[er, bad, prob1] = nntest(nn, test_xr, test_yr);
%disp(['Err-SAE',num2str(er)])
%assert(er < 0.21, 'Too big error');