clear all;
rng(400);
%%% One batch for training
[trainX, trainY, trainy] = LoadBatch('Dataset/data_batch_1.mat');
[valX, valY, valy] = LoadBatch('Dataset/data_batch_2.mat');
[d, N] = size(trainX);
[K, ~] = size(trainY);

mean_X = mean(trainX,2);
trainX = trainX - repmat(mean_X, [1, size(trainX,2)]);
valX = valX - repmat(mean_X, [1, size(valX,2)]);
%%% #################### %%%

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

function [b, W] = InitParam(m, d, K)
b1 = zeros(m,1);
b2 = zeros(K,1);
W1 = 0.001*randn(m,d);
W2 = 0.001*randn(K,m);

W = {W1, W2};
b = {b1, b2};
end

function [P, X, S] = EvaluateClassifier(X0, W, b, K)
S = {bsxfun(@plus, W{1}*X0, b{1})};
X = {max(0, S{1})};

for i=2:K+1
    S{i} = bsxfun(@plus, W{i}*X{i-1}, b{i});
    X{i} = max(0, S{i}); % ReLU activation
end

P = softmax(S{end});
end

function [b_grad, W_grad] = ComputeGradients(X, Y, W, b, lambda, GDparams)
N = size(X,2);
[P, X, S] = EvaluateClassifier(X, W, b, GDparams);
b_grad = {};
W_grad = {};
for i=1:N
    g = -(Y(:,i)-P(:,i))';
    
    for k=K:1
        b_grad{k} = g/N;
        W_grad{k} = g'*X{k}(:,i)/N + 2*lambda*W{k};
        g = g*W{k};
        g = g*diag(S{k}(:,i)>0);
    end
end

% b_grad = {dldb1/N, dldb2/N};
% W_grad = {dldw1/N + 2*lambda*W{1}, dldw2/N + 2*lambda*W{2}};
end