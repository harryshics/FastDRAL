addpath('../lib/code/');
addpath('../lib/data/');

alpha_candi = 0.1;
beta_candi = 0;
nSelInsArr = (20:20:200);

dataset = 'USPS_9298n_256d_10c';
f = FastDRAL_single(dataset,nSelInsArr,alpha_candi,beta_candi);

dataset = 'MNIST_10000n_784d_10c';
f = FastDRAL_single(dataset,nSelInsArr,alpha_candi,beta_candi);

dataset = 'news4a_3840n_4989d_4c';
f = FastDRAL_single(dataset,nSelInsArr,alpha_candi,beta_candi);