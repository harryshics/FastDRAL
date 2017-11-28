function [U,V] = rk_svd(R,k)
alpha = 0.01;
beta = 0.01;
[~,nItem] = size(R);
maxIter = 10;
iter = 0;
V = rand(k,nItem);
while iter < maxIter
    U = (V*V'+alpha*eye(k))\(R*V')';
    V = (U*U'+beta*eye(k))\(U*R);
    iter = iter + 1;
end
U = U';
V = V';
end

function obj = getObj(R,U,V,alpha,beta)
term_1 = R - U'*V;
obj = sum(sum(term_1.*term_1)) + alpha*sum(sum(U.*U)) + beta*sum(sum(V.*V));
end