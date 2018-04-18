clear all;
rng(400);
%%% One batch for training
[trainX, trainY, trainy] = LoadBatch('Dataset/data_batch_1.mat');
[valX, valY, valy] = LoadBatch('Dataset/data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('Dataset/test_batch.mat');
[d, N] = size(trainX);
[K, ~] = size(trainY);
M = [50, 30]; % number of hidden nodes

mean_X = mean(trainX,2);
trainX = trainX - repmat(mean_X, [1, size(trainX,2)]);
valX = valX - repmat(mean_X, [1, size(valX,2)]);
%%% #################### %%%

%%% Multiple batches for final test
% [tx1, tY1, ty1] = LoadBatch('Dataset/data_batch_1.mat');
% [tx2, tY2, ty2] = LoadBatch('Dataset/data_batch_2.mat');
% [tx3, tY3, ty3] = LoadBatch('Dataset/data_batch_3.mat');
% [tx4, tY4, ty4] = LoadBatch('Dataset/data_batch_4.mat');
% [tx5, tY5, ty5] = LoadBatch('Dataset/data_batch_5.mat');
% [X_test, Y_test, y_test] = LoadBatch('Dataset/test_batch.mat');
% 
% 
% X_train = [tx1, tx2, tx3, tx4, tx5(:, 1:9000)];
% Y_train = [tY1, tY2, tY3, tY4, tY5(:, 1:9000)];
% y_train = [ty1, ty2, ty3, ty4, ty5(:, 1:9000)];
% 
% mean_X_train = mean(X_train, 2);
% X_train = X_train - repmat(mean_X_train, [1, size(X_train,2)]);
% 
% X_valid = tx5(:,9001:10000);
% X_valid = X_valid - repmat(mean_X_train, [1, size(X_valid,2)]);
% Y_valid = tY5(:,9001:10000);
% y_valid = ty5(:,9001:10000);
% 
% X_test = X_test - repmat(mean_X_train, [1, size(X_test,2)]);
%%% #################### %%%

eta_opt = 0.429684777550084;
lambda_opt = 6.177962194101894e-05;

% lambda = lambda_opt;
GDparams.n_batch=100;
GDparams.rho=0.90; %momentum
GDparams.decay=0.90; % Learning rate decay
GDparams.n_epochs = 10;
GDparams.eta = eta_opt;
GDparams.alpha = 0.99;
lambda = lambda_opt;
n_runs = 100;
M = [50, 30];


% [b_grad, W_grad] = ComputeGradients(trainX(:,100), trainY(:, 100), W, b, lambda);
% CompareGradients(trainX(1:300, 1:3), trainY(:, 1:3), M);

[b, W] = InitParam(M, trainX, trainY);
e_range = {log10(0.05), log10(0.6)};
l_range = {log10(0.000001), log10(0.1)};

% Fine search range
% e_range = {log10(0.4), log10(0.5)};
% l_range = {log10(0.5e-04), log10(1.5e-04)};
params = HyperParamSearch(e_range, l_range, trainX, trainY, valX, valY, valy, X_test, y_test, GDparams, n_runs);
[I, M] = max(params(:,3))
save('storeMatrix.mat','params');

[b, W] = InitParam(M, X_train, Y_train);
[Wstar, bstar, muSAv, varSAv, tL_saved, vL_saved] = MiniBatchGD(X_train, Y_train, X_valid, Y_valid, GDparams, W, b, lambda);
acc = ComputeAccuracy(X_test, y_test, Wstar, bstar, muSAv, varSAv)
figure;
plot(tL_saved); hold on;
plot(vL_saved);
title("Cross Entropy Loss for Training and Valdidation Data");
xlabel("Epochs");
ylabel("Cross entropy loss");
legend("Training loss", "Validation loss");
% fnameMontage = sprintf('train_val_loss_leak_eta_%f_lambda_%f.png', eta_opt, lambda_opt);
% saveas(gcf, fnameMontage, 'png');
%%% Subfunctions

function params = HyperParamSearch(e_range, l_range, trainX, trainY, valX, valY, valy, testX, testy, GDparams, n_runs)
for i=1:n_runs
    disp("Starting search run:"+ num2str(i));
%     [d, N] = size(trainX);
%     [K, ~] = size(trainY);
    m = [50, 30]; % number of hidden nodes
    
    [b, W] = InitParam(m, trainX, trainY);
    
    e= e_range{1} + (e_range{2} - e_range{1})*rand(1, 1);
    eta = 10^e;
    GDparams.eta = eta;
    
    l = l_range{1} + (l_range{2} - l_range{1})*rand(1, 1);
    lambda = 10^l;
    
    [Wstar, bstar, muSAv, varSAv] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda);
    
    acc = ComputeAccuracy(valX, valy, Wstar, bstar, muSAv, varSAv);
    disp("Accuracy: " + num2str(acc));
    params(i, 1) = eta;
    params(i, 2) = lambda;
    params(i, 3) = acc;
end
end

function [X, Y, y] = LoadBatch(filename)
dataSet = load(filename);
X = double(dataSet.data)'/255;
y = double(dataSet.labels+1)';
N = length(y);
K = max(y);
Y = zeros(K, N);
for i = 1:N
    Y(y(i), i) = 1;
end
end

function [b, W] = InitParam(M, X, Y)
[d, ~] = size(X);
[L, ~] = size(Y);
K = length(M)+1;
M = [d, M, L];

W = cell(1, K); b=cell(1,K);
for k=1:K
    W{k} = 0.001*randn(M(k+1), M(k));
    b{k} = zeros(M(k+1), 1);
end
end

function [P, H, S, S_hat, muS, varS] = EvaluateClassifier(X0, W, b, varargin)
K = length(W);
N = size(X0, 2);
S = cell(1, K); S_hat = S;
H = cell(1, K);
muS = cell(1, K);
varS = cell(1, K);

S{1} = bsxfun(@plus, W{1}*X0, b{1});
if ~isempty(varargin)
    muS = varargin{1};
    varS = varargin{2};
else
    muS{1} = (1/N)*sum(S{1}, 2);
    varS{1} = var(S{1}, 0, 2)*((N-1)/N);
end
S_hat{1} = BatchNormalize(S{1}, muS{1}, varS{1});
H{1} = max(0, S_hat{1});
for i=2:K-1
    S{i} = bsxfun(@plus, W{i}*H{i-1}, b{i});
    if isempty(varargin)
        muS{i} = (1/N)*sum(S{i}, 2);
        varS{i} = var(S{i}, 0, 2)*((N-1)/N);
    end
    S_hat{i} = BatchNormalize(S{i}, muS{i}, varS{i});
    H{i} = max(0, S_hat{i});
end
S{end} = bsxfun(@plus, W{end}*H{end-1}, b{end});
P = softmax(S{end});
end

function S_hat = BatchNormalize(S, muS, varS)
epsilon = 0.001;
S_hat = ((diag(varS+epsilon))^(-0.5))*(S-muS);
end

function g = BatchNormBackPass(g, S, muS, varS, N)
epsilon = 0.001;
Vb = (varS + epsilon);
dJdVb = -0.5*sum(g.*Vb.^(-3/2).*(S-muS), 2);
dJdmub = -sum(g.*Vb.^(-1/2), 2);
dJdS = g.*Vb.^(-1/2) + (2/N)*dJdVb.*(S-muS) + dJdmub*(1/N);
g = dJdS;
end

function J = ComputeCost(X, Y, W, b, lambda, varargin)
if ~isempty(varargin)
    P = EvaluateClassifier(X, W, b, varargin{1}, varargin{2});
else
    P = EvaluateClassifier(X, W, b);
end
K = length(W);
D = size(X, 2);
Wij = 0;
for k=1:K
    Wij = Wij + sum(sum(W{k}.^2));
end
lcross = -log(sum(Y.*P));
J = (1/D)*sum(lcross)+lambda*Wij;
end

function acc = ComputeAccuracy(X, y, W, b, varargin)
if ~isempty(varargin)
    P = EvaluateClassifier(X, W, b, varargin{1}, varargin{2});
else
    P = EvaluateClassifier(X, W, b);
end

[~, kStar] = max(P);
correct = kStar==y;
acc = sum(correct)/length(correct);
end

function [b_grad, W_grad, muS, varS] = ComputeGradients(X, Y, W, b, lambda)
N = size(X,2);
K = length(W);
[P, H, S, S_hat, muS, varS] = EvaluateClassifier(X, W, b);

dJdb = cell(1,K);
dJdW = cell(1,K);

g = -(Y-P);
dJdb{K} = (1/N)*sum(g, 2);
dJdW{K} = (1/N)*g*H{K-1}' + 2*lambda*W{K};
g = W{K}'*g;
g = g.*(S_hat{K-1}>0);
for l = K-1:-1:1
    g = BatchNormBackPass(g, S{l}, muS{l}, varS{l}, N);
    dJdb{l} = (1/N)*sum(g, 2);
    if l == 1
        dJdW{l} = (1/N)*g*X' + 2*lambda*W{l};
    end
    if l > 1
        dJdW{l} = (1/N)*g*H{l-1}' + 2*lambda*W{l};
        g = W{l}'*g;
        g = g.*(S_hat{l-1}>0);
    end
end

b_grad = dJdb;
W_grad = dJdW;
end

function [Wstar, bstar, muSAv, varSAv, tL_saved, vL_saved] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda)
n_batch = GDparams.n_batch;
eta = GDparams.eta;
n_epochs = GDparams.n_epochs;
rho = GDparams.rho;
decay = GDparams.decay;
alpha = GDparams.alpha;

K = length(W);
N = size(trainX,2);
tL_saved=[];
vL_saved=[];
W_mom = cell(1, K);
b_mom = cell(1, K);

for k=1:K
    W_mom{k} = zeros(size(W{k}));
    b_mom{k} = zeros(size(b{k}));
end

disp("Original training loss: " + num2str(ComputeCost(trainX, trainY, W, b, lambda)));
for i=1:n_epochs
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = trainX(:, inds);
        Ybatch = trainY(:, inds);
        
        [b_grad, W_grad, muS, varS] = ComputeGradients(Xbatch, Ybatch, W, b, lambda);
        if i == 1 && j == 1 
           muSAv = muS;
           varSAv = varS;
        end
        for k=1:K
            muSAv{k} = alpha*muSAv{k} + (1-alpha)*muS{k};
            varSAv{k} = alpha*varSAv{k} + (1-alpha)*varS{k};
            W_mom{k} = rho*W_mom{k} + eta*W_grad{k};
            b_mom{k} = rho*b_mom{k} + eta*b_grad{k};
            W{k} = W{k} - W_mom{k};
            b{k} = b{k} - b_mom{k};
        end
    end
    eta = decay*eta;
    trainLoss = ComputeCost(trainX, trainY, W, b, lambda, muSAv, varSAv);
    disp("Current training loss: " + num2str(trainLoss));
    tL_saved = [tL_saved;trainLoss];
    valLoss = ComputeCost(valX, valY, W, b, lambda, muSAv, varSAv);
    vL_saved = [vL_saved; valLoss];
end
Wstar = W;
bstar = b;
end

function CompareGradients(trainX, trainY, M)
lambda=0;
size(trainX), size(trainY)
[b, W] = InitParam(M, trainX, trainY);
[b_grad, W_grad] = ComputeGradients(trainX, trainY, W, b, lambda);
[b_gradn, W_gradn] = ComputeGradsNumSlow(trainX, trainY, W, b, lambda, 1e-5);
% [b_gradn_quick, W_gradn_quick] = ComputeGradsNum(trainX(1:300, 1), trainY(:,1), W, b, lambda, 1e-5);
K = length(W);
for k=1:K
    w_grad_diff_slow{k} = max(max(abs(W_grad{k}-W_gradn{k})));
    b_grad_diff_slow{k} = max(max(abs(b_grad{k}-b_gradn{k})));
end
w_grad_diff_slow
b_grad_diff_slow
end

%%% Numerical gradients
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

c = ComputeCost(X, Y, W, b, lambda);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b{j}(i) = (c2-c) / h;
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
        
        grad_W{j}(i) = (c2-c) / h;
    end
end
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
        
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end



