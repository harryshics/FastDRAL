addpath('../lib/code/');
addpath('../lib/data/');

alpha_candi = 10.^(-1);
beta_candi = 10.^(-1);
nSelInsArr = (20:20:200);

dataset = 'COIL20_1440n_1024d_20c';
f = FastDRAL_single(dataset,nSelInsArr,alpha_candi,beta_candi);
