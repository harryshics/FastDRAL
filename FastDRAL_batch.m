addpath('../lib/code/');
addpath('../lib/data/');

alpha_candi = 10.^(-1);
beta_candi = 10.^(-1);
nSelInsArr = (20:20:200);

dataset = 'MNIST_10000n_784d_10c';
f = FastDRAL_single(dataset,nSelInsArr,alpha_candi,beta_candi);
% 
% dataset = 'USPS_9298n_256d_10c';
% f = FastDRAL_single(dataset,nSelInsArr,alpha_candi,beta_candi);
% 
% dataset = 'webkb_4199n_1000d_4c_tfidf';
% f = FastDRAL_single(dataset,nSelInsArr,alpha_candi,beta_candi);
% 
% dataset = 'news4a_3840n_4989d_4c';
% f = FastDRAL_single(dataset,nSelInsArr,alpha_candi,beta_candi);

% dataset = 'cifar_10_data_batch_1';
% f = FastDRAL_single(dataset,nSelInsArr,alpha_candi,beta_candi);
