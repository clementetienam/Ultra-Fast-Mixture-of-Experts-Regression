%%
clc;
clear ;
close all;
disp('@Author: Dr Clement Etienam')
disp('Prediction script')
%% Begin Programme
disp('*******************BEGIN PROGRAMME*********************************')
oldfolder=cd;
cd(oldfolder);
fdd='Prediction_folder';
mkdir(fdd);
f='MLSL_machine';
cd(f);
combo1=load('combo.out');
cd(oldfolder)

switch combo1
    case 1
addpath('CKS');
mydir = fileparts (mfilename ('fullpath'));                 
addpath (mydir)
dirs = {'cov','doc','inf','lik','mean','prior','util'};           
for d = dirs, addpath (fullfile (mydir, d{1})), end
dirs = {{'util','minfunc'},{'util','minfunc','compiled'}};     
for d = dirs, addpath (fullfile (mydir, d{1}{:})), end
addpath([mydir,'/util/sparseinv'])
addpath('Data');
cd(f)
Classifiers=load('Classifiers'); 
Classifiers=Classifiers.Classifiers;
Experts=load('Experts.out');
clfysses=load('clfysses'); 
clfysses=clfysses.clfysses;
Regressors=load('Regressors');
Regressors=Regressors.Regressors;  
Xtrainbig=load('Xtrainbig');
Xtrainbig=Xtrainbig.Xtrainbig;
ytrainbig=load('ytrainbig');
ytrainbig=ytrainbig.ytrainbig;
cd(oldfolder)
    case 2
addpath('CKS_DNN');
addpath('Data');
cd(f)
Classifiers=load('Classifiers'); 
Classifiers=Classifiers.Classifiers;
Experts=load('Experts.out');
Classallsbig=load('Classallsbig');
Classallsbig=Classallsbig.Classallsbig;
clfysses=load('clfysses'); 
clfysses=clfysses.clfysses;
Regressors=load('Regressors');
Regressors=Regressors.Regressors; 
cd(oldfolder)
    case 3
addpath('RFS');
addpath('Data');
cd(f)
Classifiers=load('Classifiers'); 
Classifiers=Classifiers.Classifiers;
Experts=load('Experts.out');
Classallsbig=load('Classallsbig');
Classallsbig=Classallsbig.Classallsbig;
clfysses=load('clfysses'); 
clfysses=clfysses.clfysses;
Regressors=load('Regressors');
Regressors=Regressors.Regressors; 
cd(oldfolder)
end
addpath('Crucial')
%% Read Training data
% C=xlsread('ALLL.xlsx');
load('jm_data.mat')
yb=[ptotped, betanped, wped];
X_test2=[r a kappa delta bt ip neped betan zeffped];
%X_test2=C(:,1:8);
for i=1:size(X_test2,2)
meanss(:,i)=max(X_test2(:,i));
end

for i=1:size(X_test2,2)
meanss2(:,i)=min(X_test2(:,i));
end
cd(f)
clfxsses=load('clfxsses'); 
clfxsses=clfxsses.clfxsses;
cd(oldfolder)
% y_train=C(:,9:10);
y_train=yb;

%%
 sd=1;
 rng(sd); % set random number generator with seed sd
 %% For now Lets use X_test2 for prediction
[Hardmean,Softmean]=Forwarding(X_test2,f,...
    clfxsses{1,1},y_train,oldfolder,combo1);

cd(fdd)
headers=cell(1,size(Hardmean,2));
for jj=1:size(Hardmean,2)
    headers(1,jj)=cellstr(strcat('Y', sprintf('%d',jj)));
end

Namefile1= 'Hard_prediction.csv';
csvwrite_with_headers(  Namefile1,Hardmean,headers);
Namefile2= 'Soft_prediction.csv';
csvwrite_with_headers(  Namefile2,Softmean,headers);
cd(oldfolder);