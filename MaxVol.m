function col_idxs = MaxVol(X,k)
%% column selection by the MaxVol algorithm
%
% X: a data matrix
% k: number of columns to be selected
%
% col_idxs: the idxs of selected columns
%
% Written by Lei Shi (harryshi.cs@gmail.com)
% Version 1 on Dec 29, 2016
% Version 2 on Nov 28, 2017

%% Compute the rank-k svd approximation to X
[~,V] = rk_svds(X,k);

%% Column selection by MaxVol
flag = 1;
p = randperm(size(V,1));
iter = 1;
while flag == 1
    B = V(p,:)*V(p(1:k),:)^(-1);
    abs_B = abs(B);
    [~,max_row_idx] = max(abs_B);
    [~,max_col_idx] = max(max(abs_B));
    row_idx = max_row_idx(max_col_idx);
    col_idx = max_col_idx;
    b_ij = abs_B(row_idx,col_idx);
    if b_ij-1 > 0.0001
       tmp = p(row_idx);
       p(row_idx) = p(col_idx);
       p(col_idx) = tmp;
    else
        flag = 0;
    end
    iter = iter + 1;
    if iter > 1000
        break;
    end
end
col_idxs = p(1:k);
end