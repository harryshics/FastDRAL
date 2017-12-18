function [SelIdx, X] = FastDRAL(X, k, alpha, beta, maxIter, options)
%% FastDRAL
% Written by Lei Shi (harryshi.cs@gmail.com)
% Version 1, Nov 26th, 2017
%
% min_{A,V,B} ||X - VA||^2 + \alpha ||A||^2 + \beta ||V-XB^T||^2
% s.t. b_{ij} \in {0,1}, B_{i.} 1 = 1
%
% X: d by n data matrix, whose columns correspond to samples and rows to features
% k: number of selected samples
% alpha, beta: hyper parameters
% maxIter: number of maximum iterations
%

[nFea, nSmp] = size(X);
V = rand(nFea, k); % V: d by k virtual samples
A = rand(k, nSmp); % A: k by n reconstruction coefficients
B = zeros(k, nSmp); % B: nearest neighbour assignment matrix
% B_old = zeros(size(B));

if nargin < 6
    init = 0;
    verbose = 1;
    reducedDim = -1;
else
    init = options.init;
    verbose = options.verbose;
    reducedDim = options.reducedDim;
end

if reducedDim > 0
    fprintf('Reduce dim to %d ...',reducedDim);
    X = GraphEmbedding(X', reducedDim);
    X = X';
    fprintf('Done!\n');
end

if init == -1
    pre_idx = options.pre_idx;
    V = X(:, pre_idx);
end

if init == 0
    %fprintf('Init randomly...');
    t_start = clock;
    rand_idx = randperm(nSmp);
    V = X(:,rand_idx(1:k));
    t_end = clock;
    t_cost = etime(t_end,t_start);
    %fprintf('Done! exe time: %f\n',t_cost);
end

if init == 1
    fprintf('Init by kmeans....');
    t_start = clock;
    [~,center,~,~] = litekmeans(X', k);
    V = center';
    t_end = clock;
    t_cost = etime(t_end,t_start);
    fprintf('Done! exe time: %f\n',t_cost);
end

if init == 2
    fprintf('Init by MaxVol....');
    t_start = clock;
    col_idxs = MaxVol(X, k);
    V = X(:,col_idxs);
    t_end = clock;
    t_cost = etime(t_end,t_start);
    fprintf('Done! exe time: %f\n',t_cost);
end

X_squre = sum(X.*X,1);
iter = 0;
while iter < maxIter
	% Update B
	B = zeros(size(B));
    for i = 1:k
        Dis = -2*V(:,i)'*X + X_squre; % a faster calculating rule
% 		V_aug = repmat(V(:,i),1,nSmp); % less faster version
% 		Dis = sum((V_aug - X).*(V_aug - X),1);
		Near_Idx = find(Dis == min(Dis));
		B(i,Near_Idx(1)) = 1;
     end
%     if sum(sum(B ~= B_old)) ~= 0
%         disp(['B updated']);
%     end
%     B_old = B;

	% Update A
	A = (V'*V + alpha*eye(k))^(-1)*V'*X;

	% Update V
	V = X*(A' + beta*B')*(A*A' + beta*eye(k))^(-1);

    obj = GetObj(X, V, A, B, alpha, beta);
    if verbose == 1
        disp(['Iter ',num2str(iter),'=',num2str(obj)]);
    end
    if verbose == 2
        if mod(iter,20) == 0
            fprintf('.');
        end
    end
	iter = iter + 1;
end
% Update B
B = zeros(size(B));
for i = 1:k
    Dis = -2*V(:,i)'*X + X_squre; % a faster calculating rule
    Near_Idx = find(Dis == min(Dis));
    B(i,Near_Idx(1)) = 1;
end
SelIdx = zeros(k,1);
for i = 1:k
    SelIdx(i) = find(B(i,:)==1);
end
end

function obj = GetObj(X, V, A, B, alpha, beta)
term_1 = X - V*A;
obj_1 = sum(sum(term_1.*term_1));
obj_2 = sum(sum(A.*A));
term_3 = V - X*B';
obj_3 = sum(sum(term_3.*term_3));
obj = obj_1 + alpha*obj_2 + beta*obj_3;
end