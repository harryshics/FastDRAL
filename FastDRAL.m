function SelIdx = FastDRAL(X, k, alpha, beta, maxIter, Verbose)
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
V = rand(nFea, k)*255; % V: d by k virtual samples
A = rand(k, nSmp); % A: k by n reconstruction coefficients
B = zeros(k, nSmp); % B: nearest neighbour assignment matrix

iter = 0;


while iter < maxIter
	% Update B
	B = zeros(size(B));
	for i = 1:k
		V_aug = repmat(V(:,i),1,nSmp);
		Dis = sum((V_aug - X).*(V_aug - X),1);
		Near_Idx = find(Dis == min(Dis));
		B(i,Near_Idx(1)) = 1;
	end

	% Update A
	A = (V'*V + alpha*eye(k))^(-1)*V'*X;

	% Update V
	V = X*(A' + beta*B')*(A*A' + beta*eye(k))^(-1);

    obj = GetObj(X, V, A, B, alpha, beta);
    if Verbose == 1
        disp(['Iter ',num2str(iter),'=',num2str(obj)]);
    end
	iter = iter + 1;
end
% Update B
B = zeros(size(B));
for i = 1:k
    V_aug = repmat(V(:,i),1,nSmp);
    Dis = sum((V_aug - X).*(V_aug - X),1);
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