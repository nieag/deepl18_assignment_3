clear all;
rng(400);
%%% One batch for training
[trainX, trainY, trainy] = LoadBatch('Dataset/data_batch_1.mat');
[valX, valY, valy] = LoadBatch('Dataset/data_batch_2.mat');
% [d, N] = size(trainX(1:300, 1));
% [L, ~] = size(trainY(:, 1));
[d, N] = size(trainX);
[L, ~] = size(trainY);

M = [50]; % Layer sizes. d, L are input and output dims

mean_X = mean(trainX,2);
trainX = trainX - repmat(mean_X, [1, size(trainX,2)]);
valX = valX - repmat(mean_X, [1, size(valX,2)]);
%%% #################### %%%

lambda = 0;
GDparams.n_batch=100;
GDparams.rho=0.90; %momentum
GDparams.decay=0.95; % Learning rate decay
GDparams.n_epochs = 10;
GDparams.eta = 0.1;

[b, W] = InitParam(M, trainX, trainY);
[P, H, S] = EvaluateClassifier(trainX, W, b);
% J = ComputeCost(trainX, trainY, W, b, lambda);

% [b_grad, W_grad] = ComputeGradients(trainX, trainY, W, b, lambda);
% CompareGradients(trainX(1:300, 1), trainY(:,1), M);
[Wstar, bstar, tL_saved, vL_saved] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda);
acc = ComputeAccuracy(trainX, trainy, Wstar, bstar)
%%% Subfunctions

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

function [P, H, S] = EvaluateClassifier(X0, W, b)
K = length(W);
S = cell(1, K);
H = cell(1, K); 
S{1} = bsxfun(@plus, W{1}*X0, b{1});
H{1} = max(0, S{1});
for i=2:K-1
    S{i} = bsxfun(@plus, W{i}*H{i-1}, b{i});
    H{i} = max(0, S{i});
end
S{K} = bsxfun(@plus, W{K}*H{K-1}, b{K});
P = softmax(S{K});
end

function J = ComputeCost(X, Y, W, b, lambda)
P = EvaluateClassifier(X, W, b);
K = length(W);
D = size(X, 2);
Wij = 0;
for k=1:K
    Wij = Wij + sum(sum(W{k}.^2));
end
lcross = -log(sum(Y.*P));
J = (1/D)*sum(lcross)+lambda*Wij;
end

function acc = ComputeAccuracy(X, y, W, b)
P = EvaluateClassifier(X, W, b);
[~, kStar] = max(P);
correct = kStar==y;
acc = sum(correct)/length(correct);
end

function [b_grad, W_grad] = ComputeGradients(X, Y, W, b, lambda)
N = size(X,2);
K = length(W);
[P, H, S] = EvaluateClassifier(X, W, b);
b_grad = cell(1, K);
W_grad = cell(1, K);

for i=1:N
    g = (-Y(:,i)'/(Y(:,i)'*P(:,i)))*(diag(P(:,i))-(P(:,i)*P(:,i)'));
    for k=K:-1:1
        b_grad{k} = g'/N;
        if k==1
           W_grad{k} = g'*X(:,i)'/N+2*lambda*W{k};
        else
            W_grad{k} = g'*H{k-1}(:,i)'/N + 2*lambda*W{k};
            g = g*W{k};
            g = g*diag(S{k-1}(:,i)>0);         
        end
    end
end
end

function [Wstar, bstar, tL_saved, vL_saved] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda)
n_batch = GDparams.n_batch;
eta = GDparams.eta;
n_epochs = GDparams.n_epochs;
rho = GDparams.rho;
decay = GDparams.decay;

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
        
        [b_grad, W_grad] = ComputeGradients(Xbatch, Ybatch, W, b, lambda);
        
        for k=1:K
            W_mom{k} = rho*W_mom{k} + eta*W_grad{k};
            b_mom{k} = rho*b_mom{k} + eta*b_grad{k};
            W{k} = W{k} - W_mom{k};
            b{k} = b{k} - b_mom{k};
        end
    end
    eta = decay*eta;
    trainLoss = ComputeCost(trainX, trainY, W, b, lambda);
    disp("Current training loss: " + num2str(trainLoss));
    tL_saved = [tL_saved;trainLoss];
    valLoss = ComputeCost(valX, valY, W, b, lambda);
    vL_saved = [vL_saved; valLoss];
end
disp("Final training loss: " + num2str(trainLoss));
Wstar = W;
bstar = b;
end

function CompareGradients(trainX, trainY, M)
% [d, N] = size(trainX(1:300, 1));
% [L, ~] = size(trainY(:, 1));
% M = [d, 50, 30, L]; % Layer sizes. d, L are input and output dims
lambda=0;

[b, W] = InitParam(M);
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
% w1_grad_diff_quick = max(max(abs(W_grad{1}-W_gradn_quick{1})))
% w2_grad_diff_quick = max(max(abs(W_grad{2}-W_gradn_quick{2})))
% b1_grad_diff_quick = max(max(abs(b_grad{1}-b_gradn_quick{1})))
% b2_grad_diff_quick = max(max(abs(b_grad{2}-b_gradn_quick{2})))
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



