function f = FastDRAL_single(dataset,nSelInsArr,alpha_candi,beta_candi)
%%
% dataset : the path of dataset
% nSelInsArr : an array which stores the numbers of selected instances, e.g.,
% [10,20,30]

%% setup
maxIter = 20;
[fea,gnd] = loadData(dataset,-1);
[nSmp,nFea] = size(fea);
nClass = length(unique(gnd));
disp(['Dataset:', dataset, ' | nSmp:',num2str(nSmp), ' | nFea:', num2str(nFea), ' | nClass:', num2str(nClass)]);
%% ACC
ACC_val = zeros(length(nSelInsArr),1);
ACC_knn_val = zeros(length(nSelInsArr),1);
ACC_te = zeros(length(nSelInsArr),1);
ACC_knn_te = zeros(length(nSelInsArr),1);
%% ROC
ROC_val = zeros(length(nSelInsArr),1);
ROC_knn_val = zeros(length(nSelInsArr),1);
ROC_te = zeros(length(nSelInsArr),1);
ROC_knn_te = zeros(length(nSelInsArr),1);

%% F_macro
F_macro_val = zeros(length(nSelInsArr),1);
F_macro_knn_val = zeros(length(nSelInsArr),1);
F_macro_te = zeros(length(nSelInsArr),1);
F_macro_knn_te = zeros(length(nSelInsArr),1);

%% F_micro
F_micro_val = zeros(length(nSelInsArr),1);
F_micro_knn_val = zeros(length(nSelInsArr),1);
F_micro_te = zeros(length(nSelInsArr),1);
F_micro_knn_te = zeros(length(nSelInsArr),1);

%% prepare parameters
paras_cnt = length(alpha_candi)*length(beta_candi);
paras = cell(paras_cnt,1);
para_idx = 1;
for alpha = alpha_candi
    for beta = beta_candi
        paras{para_idx}.alpha = alpha;
        paras{para_idx}.beta = beta;
        para_idx = para_idx + 1;
    end
end

for iSelIns = 1:length(nSelInsArr)
    k = nSelInsArr(iSelIns);
    disp(['************************************']);
    disp(['Selecting ', num2str(k), ' samples...']);
    t_fs = zeros(length(paras),1);
    result = cell(length(paras),1);
    for para_idx = 1:length(paras)
        alpha = paras{para_idx}.alpha;
        beta = paras{para_idx}.beta;
        options = [];
        options.init = 1;
        options.verbose = 2;
        t_start = clock;
        SelIdx = FastDRAL(fea', k, alpha, beta, maxIter, options);
        t_end = clock;
        t_fs(para_idx) = etime(t_end,t_start);
        result{para_idx}.InsIdx = SelIdx;
        disp(['alpha=',num2str(alpha),',beta=',num2str(beta),',time=',num2str(t_fs(para_idx))]);
    end
    disp(['Selecting ',num2str(k),' samples, time cost: ',num2str(mean(t_fs))]);

    disp(['Local evaluation...']);
    %% local evaluation
    for para_idx = 1:length(paras)
        InsIdx = result{para_idx}.InsIdx;
        labeledIdx = InsIdx;
        unlabeledIdx = setdiff((1:nSmp),labeledIdx);
        trFea = fea(labeledIdx,:);
        trGnd = gnd(labeledIdx);
        teFea = fea(unlabeledIdx,:);
        teGnd = gnd(unlabeledIdx);
    %     unique_trGnd_cnt = length(unique(trGnd));
    %     disp(['unique_trGnd_cnt:',num2str(unique_trGnd_cnt)]);
        performance = svm_ova_tvt_mm(trFea, trGnd, teFea, teGnd, teFea, teGnd);
        performance_knn = KNN_Classifier_tvt_mm(trFea, trGnd, teFea, teGnd, teFea, teGnd);

       %% ACC
        if performance.acc_val > ACC_val(iSelIns)
            ACC_val(iSelIns) = performance.acc_val;
            ACC_te(iSelIns) = performance.acc_test;
        end
        if performance_knn.acc_val > ACC_knn_val(iSelIns)
            ACC_knn_val(iSelIns) = performance_knn.acc_val;
            ACC_knn_te(iSelIns) = performance_knn.acc_test;
        end

       %% ROC
        if performance.ROC_val > ROC_val(iSelIns)
            ROC_val(iSelIns) = performance.ROC_val;
            ROC_te(iSelIns) = performance.ROC_test;
        end
        if performance_knn.ROC_val > ROC_knn_val(iSelIns)
            ROC_knn_val(iSelIns) = performance_knn.ROC_val;
            ROC_knn_te(iSelIns) = performance_knn.ROC_test;
        end

       %% F_macro
        if performance.F_macro_val > F_macro_val(iSelIns)
            F_macro_val(iSelIns) = performance.F_macro_val;
            F_macro_te(iSelIns) = performance.F_macro_test;
        end
        if performance_knn.F_macro_val > F_macro_knn_val(iSelIns)
            F_macro_knn_val(iSelIns) = performance_knn.F_macro_val;
            F_macro_knn_te(iSelIns) = performance_knn.F_macro_test;
        end

       %% F_micro
        if performance.F_micro_val > F_micro_val(iSelIns)
            F_micro_val(iSelIns) = performance.F_micro_val;
            F_micro_te(iSelIns) = performance.F_micro_test;
        end
        if performance_knn.F_micro_val > F_micro_knn_val(iSelIns)
            F_micro_knn_val(iSelIns) = performance_knn.F_micro_val;
            F_micro_knn_te(iSelIns) = performance_knn.F_micro_test;
        end
    end
    disp(['Selecting ',num2str(k),' samples, ACC=',num2str(ACC_te(iSelIns)),', ROC=',num2str(ROC_te(iSelIns))]);
end

result_path = strcat('../plot_results/','acc_',dataset,'_FastDRAL_kInit','_best','.mat');
save(result_path,'nSelInsArr','ACC_te','ACC_knn_te','ROC_te','ROC_knn_te','F_macro_te','F_macro_knn_te','F_micro_te','F_micro_knn_te');
f = 1;
end