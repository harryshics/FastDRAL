function f = DFIS_local_fast_single_server(dataset,nSelInsArr,alpha_candi,beta_candi)
%%
% dataset : the path of dataset
% nSelFeaArr : an array which stores the numbers of selected features, e.g.,
% [10,20,30]
% nSelInsArr : an array which stores the numbers of selected instances, e.g.,
% [10,20,30]

%% setup
[fea,gnd] = loadData(dataset,-1);
[nSmp,nFea] = size(fea);
nClass = length(unique(gnd));
disp(['Dataset:', dataset, ' | nSmp:',num2str(nSmp), ' | nFea:', num2str(nFea)]);
%% ACC
ACC_val = zeros(length(nSelInsArr),length(nSelFeaArr));
ACC_knn_val = zeros(length(nSelInsArr),length(nSelFeaArr));
ACC_te = zeros(length(nSelInsArr),length(nSelFeaArr));
ACC_knn_te = zeros(length(nSelInsArr),length(nSelFeaArr));
%% ROC
ROC_val = zeros(length(nSelInsArr),length(nSelFeaArr));
ROC_knn_val = zeros(length(nSelInsArr),length(nSelFeaArr));
ROC_te = zeros(length(nSelInsArr),length(nSelFeaArr));
ROC_knn_te = zeros(length(nSelInsArr),length(nSelFeaArr));

%% F_macro
F_macro_val = zeros(length(nSelInsArr),length(nSelFeaArr));
F_macro_knn_val = zeros(length(nSelInsArr),length(nSelFeaArr));
F_macro_te = zeros(length(nSelInsArr),length(nSelFeaArr));
F_macro_knn_te = zeros(length(nSelInsArr),length(nSelFeaArr));

%% F_micro
F_micro_val = zeros(length(nSelInsArr),length(nSelFeaArr));
F_micro_knn_val = zeros(length(nSelInsArr),length(nSelFeaArr));
F_micro_te = zeros(length(nSelInsArr),length(nSelFeaArr));
F_micro_knn_te = zeros(length(nSelInsArr),length(nSelFeaArr));

%% prepare parameters
paras_cnt = length(alpha_candi)*length(beta_candi)*length(gamma_candi);
paras = cell(paras_cnt,1);
para_idx = 1;
for alpha = alpha_candi
    for beta = beta_candi
        for gamma = gamma_candi
            paras{para_idx}.alpha = alpha;
            paras{para_idx}.beta = beta;
            paras{para_idx}.gamma = gamma;
            para_idx = para_idx + 1;
        end
    end
end

t_fs = zeros(length(paras),1);
%% dual selection by DFIS_diverse
parfor para_idx = 1:length(paras)
    alpha = paras{para_idx}.alpha;
    beta = paras{para_idx}.beta;
    gamma = paras{para_idx}.gamma;
    t_start = clock;
    [~,A,U] = DFIS_local_fast(fea',alpha,beta,gamma,6,20,k,c);
    [~,FeaIdx] = sort(sum(A.*A,2),'descend');
    [~,InsIdx] = sort(sum(U.*U,2),'descend');
    t_end = clock;
    t_fs(para_idx) = etime(t_end,t_start);
    result{para_idx}.FeaIdx = FeaIdx;
    result{para_idx}.InsIdx = InsIdx;
    disp(['alpha=',num2str(alpha),',beta=',num2str(beta),',gamma=',num2str(gamma),',time=',num2str(t_fs(para_idx))]);
end
disp(['Total time for dual selection:',num2str(mean(t_fs))]);

%% local evaluation
for para_idx = 1:length(paras)
    FeaIdx = result{para_idx}.FeaIdx;
    InsIdx = result{para_idx}.InsIdx;
    for iSelFea = 1:length(nSelFeaArr)
        for iSelIns = 1:length(nSelInsArr)
            labeledIdx = InsIdx(1:nSelInsArr(iSelIns));
            unlabeledIdx = setdiff([1:nSmp],labeledIdx);
            trFea = fea(labeledIdx,FeaIdx(1:nSelFeaArr(iSelFea)));
            trGnd = gnd(labeledIdx);
            teFea = fea(unlabeledIdx,FeaIdx(1:nSelFeaArr(iSelFea)));
            teGnd = gnd(unlabeledIdx);
            unique_trGnd_cnt = length(unique(trGnd));
            disp(['unique_trGnd_cnt:',num2str(unique_trGnd_cnt)]);
            performance = svm_ova_tvt_mm(trFea, trGnd, teFea, teGnd, teFea, teGnd);
            performance_knn = KNN_Classifier_tvt_mm(trFea, trGnd, teFea, teGnd, teFea, teGnd);
            
           %% ACC
            if performance.acc_val > ACC_val(iSelIns,iSelFea)
                ACC_val(iSelIns,iSelFea) = performance.acc_val;
                ACC_te(iSelIns,iSelFea) = performance.acc_test;
            end
            if performance_knn.acc_val > ACC_knn_val(iSelIns,iSelFea)
                ACC_knn_val(iSelIns,iSelFea) = performance_knn.acc_val;
                ACC_knn_te(iSelIns,iSelFea) = performance_knn.acc_test;
            end
            
           %% ROC
            if performance.ROC_val > ROC_val(iSelIns,iSelFea)
                ROC_val(iSelIns,iSelFea) = performance.ROC_val;
                ROC_te(iSelIns,iSelFea) = performance.ROC_test;
            end
            if performance_knn.ROC_val > ROC_knn_val(iSelIns,iSelFea)
                ROC_knn_val(iSelIns,iSelFea) = performance_knn.ROC_val;
                ROC_knn_te(iSelIns,iSelFea) = performance_knn.ROC_test;
            end
            
           %% F_macro
            if performance.F_macro_val > F_macro_val(iSelIns,iSelFea)
                F_macro_val(iSelIns,iSelFea) = performance.F_macro_val;
                F_macro_te(iSelIns,iSelFea) = performance.F_macro_test;
            end
            if performance_knn.F_macro_val > F_macro_knn_val(iSelIns,iSelFea)
                F_macro_knn_val(iSelIns,iSelFea) = performance_knn.F_macro_val;
                F_macro_knn_te(iSelIns,iSelFea) = performance_knn.F_macro_test;
            end
            
           %% F_micro
            if performance.F_micro_val > F_micro_val(iSelIns,iSelFea)
                F_micro_val(iSelIns,iSelFea) = performance.F_micro_val;
                F_micro_te(iSelIns,iSelFea) = performance.F_micro_test;
            end
            if performance_knn.F_micro_val > F_micro_knn_val(iSelIns,iSelFea)
                F_micro_knn_val(iSelIns,iSelFea) = performance_knn.F_micro_val;
                F_micro_knn_te(iSelIns,iSelFea) = performance_knn.F_micro_test;
            end
        end
    end
end

result_path = strcat('../plot_results/','acc_',dataset,'_DFIS_local_fastp3','k',num2str(k),'c',num2str(c),'_best','.mat');
save(result_path,'nSelFeaArr','nSelInsArr','ACC_te','ACC_knn_te','ROC_te','ROC_knn_te','F_macro_te','F_macro_knn_te','F_micro_te','F_micro_knn_te');
f = 1;
end