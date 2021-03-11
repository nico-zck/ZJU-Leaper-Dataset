function [L, S] = robust_pca(X, lambda, mu, tol, max_iter)
% - X is a data matrix (of the size N x M) to be decomposed
%   X can also contain NaN's for unobserved values
% - lambda - regularization parameter, default = 1/sqrt(max(N,M))
% - mu - the augmented lagrangian parameter, default = 10*lambda
% - tol - reconstruction error tolerance, default = 1e-6
% - max_iter - maximum number of iterations, default = 1000

[M, N] = size(X);
unobserved = isnan(X);
%��ʹ��Matlab�������ʱ�������������ݲ������ֵ����������NaN���������Щ�����ǲ���ʹ�õ�,��isnan���������
%tf=isnan(A)������һ����A��ͬά�������飬��A��Ԫ��ΪNaN������ֵ�����ڶ�Ӧλ���Ϸ����߼�1���棩�����򷵻��߼�0���٣���
%������z�����z��ʵ�����鲿����NaN����ôisnan���������߼�1�����ʵ�����鲿����inf���򷵻��߼�0��
X(unobserved) = 0;
normX = norm(X, 'fro');%n=norm(A),����A���������ֵ����max(svd(A))

% default arguments
if nargin < 2%matalb �ṩ������ȡ����������Ŀ�ĺ�����nargin���غ����������������
    lambda = 1 / sqrt(max(M,N));
end
if nargin < 3
    mu = 10*lambda;
end
if nargin < 4
    tol = 1e-6;
end
if nargin < 6
    max_iter = 2000;
end

% initial solution
L = zeros(M, N);
S = zeros(M, N);
Y = zeros(M, N);

for iter = (1:max_iter)
    % ADMM step: update L and S
    L = Do(1/mu, X - S + (1/mu)*Y);%���µ��Ⱦ���
    S = So(lambda/mu, X - L + (1/mu)*Y);%����ϡ�����
    % and augmented lagrangian multiplier
    Z = X - L - S;
    Z(unobserved) = 0; % skip missing values
    Y = Y + mu*Z;
    
    err = norm(Z, 'fro') / normX;
    if (iter == 1) || (mod(iter, 10) == 0) || (err < tol)
        fprintf(1, 'iter: %04d\terr: %f\trank(L): %d\tcard(S): %d\n', ...
            iter, err, rank(L), nnz(S(~unobserved)));
    end
    if (err < tol)
        break; 
    end
end
end

function r = So(tau, X)
% shrinkage operator
r = sign(X) .* max(abs(X) - tau, 0);
end

function r = Do(tau, X)
% shrinkage operator for singular values
[U, S, V] = svd(X, 'econ');
r = U*So(tau, S)*V';
end