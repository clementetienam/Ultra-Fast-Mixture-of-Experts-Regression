%%
clc;
clear ;
close all;
disp('@Author: Dr Clement Etienam')
disp('Multi Output CCR Workflow')
disp('')
disp('*******************************************************************')
%% Construct a CCR supervised learning model
disp('Method 1: Mixture of Experts model -GP experts and DNN gate')
disp('Method 2: Mixture of Experts model-DNN experts and DNN gate')
disp('Method 3: Mixture of Experts model-RF experts and RF gate')
disp('')
disp('*******************************************************************')
%%
disp('1=Mixture of experts with Gp experts and DNN gate')
disp('2=Mixture of experts with DNN experts and DNN gate')
disp('3=Mixture of experts with RF experts and RF gate')
Ultimate_clement=input('Enter the combinations of experts and Gates desired: ');
if (Ultimate_clement > 3) || (Ultimate_clement < 1)
error('Wrong choice please select 1-3')
end
oldfolder=cd;
cd(oldfolder);
addpath('Data');
disp('*******************************************************************')
% Tbl=xlsread('ALLL.xlsx');
% X=Tbl(:,1:8);
% yb=Tbl(:,9:10);
load('jm_data.mat')
yb=[ptotped, betanped, wped];
X=[r a kappa delta bt ip neped betan zeffped];
disp('*******************************************************************')
%%
if Ultimate_clement==1
 Regressors=cell(size(yb,2),1);
 Classifiers=cell(size(yb,2),1);
 Xtrainbig= cell(size(yb,2),1);
 ytrainbig= cell(size(yb,2),1);
 Classallsbig= cell(size(yb,2),1);
 clfysses= cell(size(yb,2),1);
 clfxsses= cell(size(yb,2),1);
 Trainingsets=  cell(size(yb,2),1); 
 
elseif Ultimate_clement==2
 Regressors=cell(size(yb,2),1);
 Classifiers=cell(size(yb,2),1);
 Classallsbig= cell(size(yb,2),1);
 clfysses= cell(size(yb,2),1);
 clfxsses= cell(size(yb,2),1);
 Trainingsets=  cell(size(yb,2),1);    
    
else
  Regressors=cell(size(yb,2),1);
 Classifiers=cell(size(yb,2),1);
 Classallsbig= cell(size(yb,2),1);
 clfysses= cell(size(yb,2),1);
 clfxsses= cell(size(yb,2),1);
 Trainingsets=  cell(size(yb,2),1);       
    
end
%%
disp('*******************************************************************')
folder = strcat('MLSL_machine');
mkdir(folder);
switch Ultimate_clement
    case 1
 %%      
disp('*******************************************************************')
disp('BROAD OPTION OF FITTING A MODEL USING MIXTURE OF EXPERTS')
disp(' The experts are Gp and the Gate is a DNN')
disp('SET UP GPML TOOLBOX')
disp ('executing gpml startup script...')
mydir = fileparts (mfilename ('fullpath'));                 
addpath (mydir)
dirs = {'cov','doc','inf','lik','mean','prior','util'};           
for d = dirs, addpath (fullfile (mydir, d{1})), end
dirs = {{'util','minfunc'},{'util','minfunc','compiled'}};     
for d = dirs, addpath (fullfile (mydir, d{1}{:})), end
addpath([mydir,'/util/sparseinv'])
addpath('CKS');


%% Select the Data
disp('*******************************************************************')
Datause=13;
cd(oldfolder)
disp('*******************************************************************')
disp('SELECT OPTION FOR TRAINING THE MODEL')
disp('1:CCR')
disp('2:CCR-MM')
disp('3:MM-MM')
method=input('Enter the learning scheme desired: ');
if method > 3
error('Wrong choice please select 1-3')
end
if (method==2) || (method==3)
maxitercc=input('Enter the maximum iteration: '); 
end
for jjm=1:size(yb,2)
 fprintf('Training measurement %d | %d .\n', jjm,size(yb,2));   
y=yb(:,jjm);
%% Summary of Data
    [a,b]=size(X);
    c=size(y,1);

%% Options for Training

disp('*******************************************************************')
%% Option to select the inducing points
disp('SELECT OPTION FOR INITIALISING INDUCING POINTS')
disp('1:K-means') % This throws an error sometimes
disp('2:Random')

%method2=input('Enter the options for initialising inducing points: ');
method2=1;
if method2 > 2
error('Wrong choice please select 1-2')
end
disp('*******************************************************************')
%% Rescale data and then Split dataset to Train and Test;
Xini=X;
yini=y;
  
clfx = MinMaxScaler();
(clfx.fit(X));
X=(clfx.transform(X));

clfy = MinMaxScalery();

(clfy.fit(y));
y=(clfy.transform(y));
disp('*******************************************************************')
% Test_percentage=input('Enter the fraction of test data (in decimals) required (0.1-0.3): ');
Test_percentage=0.2;
disp('')
[X_train, X_test, y_train, y_test,ind_train,ind_test] = train_test_split...
    (X,y,Test_percentage);
%%
disp('SELECT OPTION FOR EXPERTS')
disp('1:Recommended number of Experts')
disp('2:User specific')

mummy=input('Enter the options for choosing number of experts: ');
%mummy=1;
if mummy > 2
error('Wrong choice please select 1-2')
end
disp('*******************************************************************')
%% Experts options
if mummy==1

        Experts=7;       
else
disp('*******************************************************************')
disp('SELECT OPTION FOR THE EXPERTS')
%Experts=input('Enter the maximum number of experts required: ');
Experts=20;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
%Data=[X_train size(X_train,2)*(y_train)*inflate];
Data=[clfy.inverse_transform(y_train)];
[IDX,C,SUMD,Kk]=kmeans_opt(Data,30); %Elbow method
Experts=min(Experts,Kk);
end
fprintf('The Expert that will be used is : %d \n',Experts);
disp('*******************************************************************')
%% Choices for NN classification
disp('*******************************************************************')
disp('Choices for NN classification')
disp('1:Pre-set options (As with the Paper)') % This throws an error sometimes
disp('2:User preferred options')

%choicee=input('Enter the options for setting the NN classifier parameters: ');
choicee=1;
if choicee > 2
error('Wrong choice please select 1-2')
end
if choicee==1
    nnOptions = {'lambda', 0.1,...
            'maxIter', 1000,...
            'hiddenLayers', [200 40 30],...
            'activationFn', 'sigm',...
            'validPercent', 10,...
            'doNormalize', 1};


input_layer_size  = size(X_train, 2);
nrOfLabels = Experts;
hiddenLayers = [200 40 30];
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 
initial_nn_params = randInitializeWeights(layers);
else
disp('*******************************************************************')    
maxIter= input('Enter the maximum number of epochs for the Neural Network (500-1000): ');
validPercent=0.1;
size_NN=input('Enter the number of hidden Layers you require (MLP=1,DNN=>3): ');
Nodess=input('Enter the mean number of Nodes you require for the network (50?): ');
r = abs((normrnd(Nodess,20,1,size_NN)));
r=sort(round(r));
temp=r;
temp(:,2)=r(:,end);
temp(:,end)=20*size(y_train,2);
hiddenLayers=temp;
disp('*******************************************************************')

%% Options for the Neural Network Classifier
nnOptions = {'lambda', 0.1,...
            'maxIter', maxIter,...
            'hiddenLayers', hiddenLayers,...
            'activationFn', 'sigm',...
            'validPercent', validPercent,...
            'doNormalize', 1};
input_layer_size  = size(X_train, 2);
nrOfLabels = Experts;
% hiddenLayers = hiddenLayers;
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 
initial_nn_params = randInitializeWeights(layers);

%% Choices used in the paper

end
 sd=1;
 rng(sd); % set random number generator with seed sd
%% Start Simulations for CCR,CCR-MM and MM-MM
oldfolder=cd;
cd(oldfolder) % setting original directory
if method==1
disp('*******************************************************************')    
disp('CCR SCHEME') 
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
disp('*******************************************************************')
disp('DO CLUSTERING STEP')
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);


idx = kmeans(clfy.inverse_transform(y_train),Experts,'MaxIter',500);
dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
modelNN=Classify_Clement(X_train,dd,Experts);
[dd,~]=pred_class(X_train,modelNN);
diff_c=max(y_train)-min(y_train);
Class_all=cell(Experts,1);
%% Gp parameters for experts
meanfunc=@meanConst;
likfunc = {@likGauss};    
inf = @infGaussLik;
cov = {@covSEiso}; 
infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);
disp('*******************************************************************')
disp('DO REGRESSION STEP')
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor i=1:Experts
 fprintf('Starting Expert %d... .\n', i);     
 Classe= Class_all{i,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,...
    Classe,meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{i,1}=hyper_updated;
    Xtrains{i,1}=Xuse;
    ytrains{i,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', i);     
end

tt=toc;
%% Prediction on Training data Training accuracy);
[dd_unie,~]=pred_class(X_train,modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,stdtr,costhardt]=prediction_clement(weights_updated,dd_unie,...
    X_train,y_train,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,sstdtr,costsoftt]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_train,y_train,Xtrains,ytrains,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

disp('*******************************************************************')
hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);

stdtr=clfy.inverse_transform(stdtr);
sstdtr=clfy.inverse_transform(sstdtr);
%% Prediction on Test data (Test accuracy)
[dd_unie,~] = pred_class(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test data')
disp('*******************************************************************')

[Valueehard,stdte,costhard]=prediction_clement(weights_updated,dd_unie,X_test,...
    y_test,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,sstdte,costsoft]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_test,y_test,Xtrains,ytrains,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);
stdte=clfy.inverse_transform(stdte);
sstdte=clfy.inverse_transform(sstdte);


[hardanswer,softanswer,ind_train,ind_test,stdclem,stdsclem]=Plot_perform...
    (hardtr,softtr,hardts,softts,yini,...
    method,folder,Xini,ind_train,ind_test,oldfolder,...
    Datause,stdtr,stdte,sstdtr,sstdte,jjm);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 
fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

cd(folder)
Namefile=strcat('Summary_', sprintf('%d',jjm),'.out');
file5 = fopen(Namefile,'w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');

Matrix=[hardanswer,softanswer,stdclem,stdsclem];
headers = {'Hard_pred','Soft_pred','Hard_Variance','Soft_Variance'}; 
Namefile2=strcat('output_answer_', sprintf('%d',jjm),'.csv');
csvwrite_with_headers(  Namefile2,Matrix,headers);
cd(oldfolder)

Regressors{jjm,1}=weights_updated;
Classifiers{jjm,1}=modelNN;
Xtrainbig{jjm,1}= Xtrains;
ytrainbig{jjm,1}= ytrains;
Classallsbig{jjm,1}= Class_all;
clfysses{jjm,1}= clfy;
clfxsses{jjm,1}= clfx;
Trainingsets{jjm,1}= [Xini,yini]; 
Expertsbig(jjm,:)=Experts;

elseif method==2
R2_allmm=zeros(maxitercc,1);
L2_allmm=zeros(maxitercc,1);
RMSE_allmm=zeros(maxitercc,1);
valueallmm=zeros(size(y_train,1),maxitercc);    

disp('MM SCHEME')
disp('*******************************************************************')
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);
disp('*******************************************************************')
disp('DO CLUSTERING STEP')
idx = kmeans(clfy.inverse_transform(y_train),Experts,'MaxIter',500);

dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
modelNN=Classify_Clement(X_train,dd,Experts);
[dd,~] = pred_class(X_train, modelNN); % Predicts the Labels            
diff_c=max(y_train)-min(y_train);
Class_all=cell(Experts,1);
%% Gp parameters for experts
meanfunc=@meanConst;
likfunc = {@likGauss};    
inf = @infGaussLik;
cov = {@covSEiso}; 
infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);

% a=cell(10,1); % You can initialise a cell this way also
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ij=1:Experts
 fprintf('Starting Expert %d... .\n', ij);     
 Classe= Class_all{ij,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,Classe,...
    meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{ij,1}=hyper_updated;
    Xtrains{ij,1}=Xuse;
    ytrains{ij,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', ij);     
end
disp('optimise classifier')
disp('*******************************************************************')

% [modelNN,updated_classtheta] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
[dd,~] = pred_class(X_train, modelNN); % Predicts the Labels              
[Valuee1,std1,cost3]=prediction_clement(weights_updated,dd,X_train,y_train,...
    Xtrains,ytrains,Experts);
    R2ccr=cost3.R2;
    L2ccr=cost3.L2;
   RMSEccr=cost3.RMSE;
fprintf('The R2 accuracy for 1 pass CCR is %4.2f \n',R2ccr)
fprintf('The L2 accuracy for 1 pass CCR is %4.2f \n',L2ccr)
fprintf('The root mean squared error for 1 pass CCR is %4.2f \n',RMSEccr)
disp('*******************************************************************')
R2now=R2ccr; 
%% Starting MM loop
 for i=1:Inf
fprintf('iteration %d... .\n', i); 
weights=weights_updated;
if i~=1
dd = MM_clement(weights,X_train,y_train,modelNN,Class_all,Experts); 
end
Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);

disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ik=1:Experts
 fprintf('Starting Expert %d... .\n', ik);     
 Classe= Class_all{ik,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,Classe,...
    meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{ik,1}=hyper_updated;
    Xtrains{ik,1}=Xuse;
    ytrains{ik,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', ik);     
end
           
dd_updated = MM_clement(weights_updated,X_train,y_train,modelNN,Class_all,Experts);
modelNN=Classify_Clement(X_train,dd_updated,Experts);  
 %[dd_updated,D] = pred_class(X_train, modelNN); % Predicts the Labels        
 [Valuee,~,cost2]=prediction_clement(weights_updated,dd_updated,X_train,...
     y_train,Xtrains,ytrains,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
disp('*******************************************************************') 

R2_allmm(i,1)=R2;
L2_allmm(i,1)=L2;
RMSE_allmm(i,1)=RMSE;
valueallmm(:,i)=Valuee;

fprintf('R2 went from %4.4f to %4.4f... .\n', R2now,R2);    
if abs(R2-R2now) < (0.0001) || (i==maxitercc) || (RMSE==0.00) || (R2==100)
   break;
end
R2now=R2;
    %fprintf('Finished iteration %d... .\n', i);          
 end
 %%
Class_all=cell(Experts,1);
%% Gp parameters for experts
meanfunc=@meanConst;
likfunc = {@likGauss};    
inf = @infGaussLik;
cov = {@covSEiso}; 
infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
for i=1:Experts
    Classe=find(dd_updated==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);

% a=cell(10,1); % You can initialise a cell this way also
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ij=1:Experts
 fprintf('Starting Expert %d... .\n', ij);     
 Classe= Class_all{ij,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,Classe,...
    meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{ij,1}=hyper_updated;
    Xtrains{ij,1}=Xuse;
    ytrains{ij,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', ij);     
end 
%  [modelNN,updated_classtheta] = learnNN(X_train, dd_updated, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
 %%
oldfolder=cd;
cd(oldfolder) % setting original directory
tt=toc;
geh=[RMSEccr; RMSE_allmm];
iterr=size(geh,1);
xx=1:iterr;

figure()
subplot(2,2,1)
plot(xx,[RMSEccr; RMSE_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('RMSE') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');

subplot(2,2,2)
plot(xx,[R2ccr; R2_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('R2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');

subplot(2,2,3)
plot(xx,[L2ccr; L2_allmm],'r','LineWidth',1)

xlim([1 iterr])
ylabel('L2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('CCR-MM','location','northeast');
cd(folder)
Namefilef=strcat('performance_a', sprintf('%d',jjm),'.fig');
saveas(gcf,Namefilef)
cd(oldfolder)
%% Prediction on Training data Training accuracy);
[dd_unie,~] = pred_class(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,stdtr,costhardt]=prediction_clement(weights_updated,dd_unie,...
    X_train,y_train,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,sstdtr,costsoftt]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_train,y_train,Xtrains,ytrains,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;
disp('*******************************************************************')

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
stdtr=clfy.inverse_transform(stdtr);
sstdtr=clfy.inverse_transform(sstdtr);
%% Prediction on Test data (Test accuracy)

[dd_unie,~] = pred_class(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test')
disp('*******************************************************************')
[Valueehard,stdte,costhard]=prediction_clement(weights_updated,dd_unie,X_test,...
    y_test,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on test')
disp('*******************************************************************')
[Valueesoft,sstdte,costsoft]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_test,y_test,Xtrains,ytrains,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);
stdte=clfy.inverse_transform(stdte);
sstdte=clfy.inverse_transform(sstdte);
[hardanswer,softanswer,ind_train,ind_test,stdclem,stdsclem]=Plot_perform...
    (hardtr,softtr,hardts,softts,yini,...
    method,folder,Xini,ind_train,ind_test,oldfolder,Datause,stdtr,stdte,...
    sstdtr,sstdte,jjm);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 
fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

cd(folder)
Namefile=strcat('Summary_', sprintf('%d',jjm),'.out');
file5 = fopen(Namefile,'w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');

Matrix=[hardanswer,softanswer,stdclem,stdsclem];
headers = {'Hard_pred','Soft_pred','Hard_Variance','Soft_Variance'}; 
Namefile2=strcat('output_answer_', sprintf('%d',jjm),'.csv');
csvwrite_with_headers(  Namefile2,Matrix,headers);

save(strcat('R2evolution_', sprintf('%d',jjm),'.out'),...
    'R2_allmm','-ascii')
save(strcat('L2evolution_', ...
    sprintf('%d',jjm),'.out'),'L2_allmm','-ascii')
save(strcat('RMSEevolution_', sprintf('%d',jjm),'.out'),...
    'RMSE_allmm','-ascii')
save(strcat('Valueevolution_', sprintf('%d',jjm),'.out'),...
    'valueallmm','-ascii')
cd(oldfolder)
Regressors{jjm,1}=weights_updated;
Classifiers{jjm,1}=modelNN;
Xtrainbig{jjm,1}= Xtrains;
ytrainbig{jjm,1}= ytrains;
Classallsbig{jjm,1}= Class_all;
clfysses{jjm,1}= clfy;
clfxsses{jjm,1}= clfx;
Trainingsets{jjm,1}= [Xini,yini]; 
Expertsbig(jjm,:)=Experts;
else
disp('*******************************************************************')    
  disp('random MM SCHEME') 
R2_allmm=zeros(maxitercc,1);
L2_allmm=zeros(maxitercc,1);
RMSE_allmm=zeros(maxitercc,1);
valueallmm=zeros(size(y_train,1),maxitercc);    
%  parpool('cluster1',8) 
tic;
 R2now=0; 
 meanfunc=@meanConst;
likfunc = {@likGauss};    
inf = @infGaussLik;
cov = {@covSEiso}; 
infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
%% Starting MM loop
 for i=1:Inf
fprintf('iteration %d... .\n', i); 
if i==1

 dd = randi(Experts,size(y_train,1),1);
 disp('Initialised randomly for the first time')
else
weights=weights_updated;
dd = MM_clement(weights,X_train,y_train,modelNN,Class_all,Experts); 
disp('initialised using MM scheme')
end
diff_c=max(y_train)-min(y_train);

Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);
disp('*******************************************************************')
disp('Optimise experts in parallel')

parfor il=1:Experts
 fprintf('Starting Expert %d... .\n', il);   
 
 Classe= Class_all{il,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,Classe,...
    meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{il,1}=hyper_updated;
    Xtrains{il,1}=Xuse;
    ytrains{il,1}=yuse;
 end
 %fprintf('Finished Expert %d... .\n', il);     
end

if i==1
[Valueeini,~,costini]=prediction_clement(weights_updated,dd,X_train,y_train,...
    Xtrains,ytrains,Experts);
fprintf('R2 initial accuracy for random initialisation is %4.4f... .\n', costini.R2);   
end

if i==1
dd_updated=dd;
else
dd_updated = MM_clement(weights_updated,X_train,y_train,modelNN,Class_all,Experts);
end

modelNN=Classify_Clement(X_train,dd_updated,Experts);


 [Valuee,~,cost2]=prediction_clement(weights_updated,dd_updated,X_train,...
     y_train,Xtrains,ytrains,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
R2_allmm(i,:)=R2;
L2_allmm(i,:)=L2;
RMSE_allmm(i,:)=RMSE;
valueallmm(:,i)=Valuee;
fprintf('R2 went from %4.4f to %4.4f... .\n', R2now,R2);    
%if i>=2

if (abs(R2-R2now) < 0.0001) || (i==maxitercc)
   break;
end

if (R2==100) || (RMSE==0.00) 
   break;
end
%end
R2now=R2;
    fprintf('Finished iteration %d... .\n', i);          
 end
 %%
Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd_updated==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
Xtrains=cell(Experts,1);
ytrains=cell(Experts,1);

disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ik=1:Experts
 fprintf('Starting Expert %d... .\n', ik);     
 Classe= Class_all{ik,1}; 
 if size(Classe,1)>= 2
[hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,Classe,...
    meanfunc,likfunc,inf,cov,infv,method2);
    weights_updated{ik,1}=hyper_updated;
    Xtrains{ik,1}=Xuse;
    ytrains{ik,1}=yuse;
 end
 fprintf('Finished Expert %d... .\n', ik);     
end 
%%           
oldfolder=cd;
cd(oldfolder) % setting original directory
tt=toc;
geh=[RMSE_allmm];
iterr=size(geh,1);
xx=1:iterr;
figure()
subplot(2,2,1)
plot(xx,[RMSE_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('RMSE') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM-MM','location','northeast');

subplot(2,2,2)
plot(xx,[R2_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('R2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM-MM','location','northeast');

subplot(2,2,3)
plot(xx,[L2_allmm],'r','LineWidth',1)

xlim([1 iterr])
ylabel('L2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('random-MM','location','northeast');
cd(folder)
Namefilef=strcat('performance_a', sprintf('%d',jjm),'.fig');
saveas(gcf,Namefilef)
cd(oldfolder)
%% Prediction on Training data Training accuracy);
[dd_unie,~] = pred_class(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
[Valueehardtr,stdtr,costhardt]=prediction_clement(weights_updated,dd_unie,...
    X_train,y_train,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on training data')
[Valueesoftt,sstdtr,costsoftt]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_train,y_train,Xtrains,ytrains,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
stdtr=clfy.inverse_transform(stdtr);
sstdtr=clfy.inverse_transform(sstdtr);
%% Prediction on Test data (Test accuracy)
[dd_unie,D] = pred_class(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test data')
[Valueehard,stdte,costhard]=prediction_clement(weights_updated,dd_unie,X_test,...
    y_test,Xtrains,ytrains,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,sstdte,costsoft]=Unseen_soft_prediction_clement(weights_updated,...
    modelNN,X_test,y_test,Xtrains,ytrains,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);
stdte=clfy.inverse_transform(stdte);
sstdte=clfy.inverse_transform(sstdte);
[hardanswer,softanswer,ind_train,ind_test,stdclem,stdsclem]=Plot_perform...
    (hardtr,softtr,hardts,softts,yini,...
    method,folder,Xini,ind_train,ind_test,oldfolder,Datause,stdtr,stdte,...
    sstdtr,sstdte,jjm);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 
fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

cd(folder)

Namefile=strcat('Summary_', sprintf('%d',jjm),'.out');
file5 = fopen(Namefile,'w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');
save(strcat('R2evolution_', sprintf('%d',jjm),'.out'),...
    'R2_allmm','-ascii')
save(strcat('L2evolution_', ...
    sprintf('%d',jjm),'.out'),'L2_allmm','-ascii')
save(strcat('RMSEevolution_', sprintf('%d',jjm),'.out'),...
    'RMSE_allmm','-ascii')
save(strcat('Valueevolution_', sprintf('%d',jjm),'.out'),...
    'valueallmm','-ascii')
Matrix=[hardanswer,softanswer,stdclem,stdsclem];
headers = {'Hard_pred','Soft_pred','Hard_variance','Soft_Variance'}; 
Namefile2=strcat('output_answer_', sprintf('%d',jjm),'.csv');
csvwrite_with_headers(  Namefile2,Matrix,headers);


ind_train=reshape(ind_train,[],1);
ind_test=reshape(ind_test,[],1);
save('Train_indices.out','ind_train','-ascii')
save('Test_indices.out','ind_test','-ascii')
cd(oldfolder)

Regressors{jjm,1}=weights_updated;
Classifiers{jjm,1}=modelNN;
Xtrainbig{jjm,1}= Xtrains;
ytrainbig{jjm,1}= ytrains;
Classallsbig{jjm,1}= Class_all;
clfysses{jjm,1}= clfy;
clfxsses{jjm,1}= clfx;
Trainingsets{jjm,1}= [Xini,yini]; 
Expertsbig(jjm,:)=Experts;
cd(oldfolder) 
end
end
disp('*******************************************************************')
cd(folder)
parsave2(Regressors,...
    Classifiers,Classallsbig,clfysses,clfxsses,...
   Xtrainbig,ytrainbig,Trainingsets)
save('combo.out','Ultimate_clement','-ascii')
save('Experts.out','Expertsbig','-ascii')
cd(oldfolder)
rmpath('CKS')
rmpath(mydir)
rmpath('data')
%end   
    case 2
   %%
disp('*******************************************************************')
disp('BROAD OPTION OF FITTING A MODEL USING MIXTURE OF EXPERTS')
disp('*******************************************************************')
disp(' The experts are DNN and the Gate is a DNN')
addpath('CKS_DNN');
addpath('Data');
oldfolder=cd;

%% Select the Data to use
disp('*******************************************************************')
disp('*******************************************************************')
disp('SELECT OPTION FOR TRAINING THE MODEL')
disp('1:CCR')
disp('2:CCR-MM')
disp('3:MM-MM')
method=input('Enter the learning scheme desired: ');
if method > 3
error('Wrong choice please select 1-3')
end
if (method==2) || (method==3)
maxitercc=input('Enter the maximum iteration: '); 
%  maxitercc=20;  
end

Datause=10;
 sd=1;
cd(oldfolder)
for jjm=1:size(yb,2)
 fprintf('measurement %d | %d .\n', jjm,size(yb,2));  
 y=yb(:,jjm);
%% Summary of Data
 
    [a,b]=size(X);
    c=size(y,1);

%% Options for Training

disp('*******************************************************************')
%% Split dataset to Train and Test;    
Xini=X;
yini=y;
  
clfx = MinMaxScaler();
(clfx.fit(X));
X=(clfx.transform(X));

clfy = MinMaxScalery();
(clfy.fit(y));
y=(clfy.transform(y));
disp('*******************************************************************')
% Test_percentage=input('Enter the fraction of test data (in decimals) required (0.1-0.3): ');
Test_percentage=0.2;
disp('')
[X_train, X_test, y_train, y_test,ind_train,ind_test] = train_test_split...
    (X,y,Test_percentage);
%%
disp('SELECT OPTION FOR EXPERTS')
disp('1:Recommended number of Experts') % This throws an error sometimes
disp('2:User specific')

%mummy=input('Enter the options for choosing number of experts: ');
mummy=2;
if mummy > 2
error('Wrong choice please select 1-2')
end
disp('*******************************************************************')
%% Experts options
if mummy==1
        Experts=7;        
else
disp('*******************************************************************')
disp('SELECT OPTION FOR THE EXPERTS')
%Experts=input('Enter the maximum number of experts required: ');
Experts=20;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
%Data=y_train;
[IDX,C,SUMD,Kk]=kmeans_opt(Data,20); %Elbow method
Experts=min(Experts,Kk);
end
fprintf('The Expert that will be used is : %d \n',Experts);
disp('*******************************************************************')
%% Choices for NN classification
disp('*******************************************************************')
disp('Choices for NN classification')
disp('1:Pre-set options (As with the Paper)') 
disp('2:User prefered options')

%choicee=input('Enter the options for setting the NN classifier parameters: ');
choicee=1;
if choicee > 2
error('Wrong choice please select 1-2')
end
if choicee==1
    nnOptions = {'lambda', 0.1,...
            'maxIter', 1000,...
            'hiddenLayers', [200 40 30],...
            'activationFn', 'sigm',...
            'validPercent', 10,...
            'doNormalize', 1};


input_layer_size  = size(X_train, 2);
nrOfLabels = Experts;
hiddenLayers = [200 40 30];
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 
initial_nn_params = randInitializeWeights(layers);
else
maxIter= input('Enter the maximum number of epochs for the Neural Network (500-1000): ');
validPercent=0.1;
size_NN=input('Enter the number of hidden Layers you require (MLP=1,DNN=>3): ');
Nodess=input('Enter the mean number of Nodes you require for the network (50?): ');
r = abs((normrnd(Nodess,20,1,size_NN)));
r=sort(round(r));
temp=r;
temp(:,2)=r(:,end);
temp(:,end)=20*size(y_train,2);
hiddenLayers=temp;
disp('*******************************************************************')

%% Options for the Neural Network Claasifier
nnOptions = {'lambda', 0.1,...
            'maxIter', maxIter,...
            'hiddenLayers', hiddenLayers,...
            'activationFn', 'sigm',...
            'validPercent', validPercent,...
            'doNormalize', 1};
input_layer_size  = size(X_train, 2);
nrOfLabels = Experts;
% hiddenLayers = hiddenLayers;
layers = [input_layer_size, hiddenLayers, nrOfLabels]; 
initial_nn_params = randInitializeWeights(layers);
end

%% Start Simuations for CCR,CCR-MM and MM-MM
oldfolder=cd;
cd(oldfolder) % setting original directory
if method==1
disp('*******************************************************************')    
disp('CCR SCHEME') 
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
disp('*******************************************************************')
disp('DO CLUSTERING STEP')
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);
idx = kmeans(y_train,Experts,'MaxIter',500);
dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
modelNN=Classify_Clement(X_train,dd,Experts);
[dd,~]=pred_class(X_train,modelNN);
diff_c=max(y_train)-min(y_train);
Class_all=cell(Experts,1);
%% 

for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);
disp('*******************************************************************')
disp('DO REGRESSION STEP')
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor i=1:Experts
 fprintf('Starting Expert %d... .\n', i);     
 Classe= Class_all{i,1}; 
 if size(Classe,1)~= 0
[net]=optimise_experts_dnn_2(X_train,y_train,Classe);
 weights_updated{i,1}=net;

 end
 fprintf('Finished Expert %d... .\n', i);     
end

tt=toc;
%% Prediction on Training data Training accuracy);
[dd_unie,~] = pred_class(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,costhardt]=prediction_clement_dnn_2...
    (weights_updated,dd_unie,X_train,y_train,Class_all,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement_dnn_2...
    (weights_updated,modelNN,X_train,y_train,Class_all,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

disp('*******************************************************************')
hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)
[dd_unie,D] = pred_class(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test data')
disp('*******************************************************************')
[Valueehard,costhard]=prediction_clement_dnn_2...
    (weights_updated,dd_unie,X_test,y_test,Class_all,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement_dnn_2...
    (weights_updated,modelNN,X_test,y_test,Class_all,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);

[hardanswer,softanswer]=Plot_perform_dnn(hardtr,softtr,hardts,softts,...
    yini,method,folder,Xini,ind_train,ind_test,oldfolder,Datause,jjm);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 

fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

cd(folder)
Namefile=strcat('Summary_', sprintf('%d',jjm),'.out');
file5 = fopen(Namefile,'w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');
Matrix=[hardanswer,softanswer];
headers = {'Hard_pred','Soft_pred'}; 
Namefile2=strcat('output_answer_', sprintf('%d',jjm),'.csv');
csvwrite_with_headers(  Namefile2,Matrix,headers);

cd(oldfolder)

Regressors{jjm,1}=weights_updated;
Classifiers{jjm,1}=modelNN;
Classallsbig{jjm,1}= Class_all;
clfysses{jjm,1}= clfy;
clfxsses{jjm,1}= clfx;
Trainingsets{jjm,1}= [Xini,yini]; 
Expertsbig(jjm,:)=Experts;


elseif method==2
R2_allmm=zeros(maxitercc,1);
L2_allmm=zeros(maxitercc,1);
RMSE_allmm=zeros(maxitercc,1);
valueallmm=zeros(size(y_train,1),maxitercc);    
   
disp('-------------------------MM SCHEME----------------------------')
disp('*******************************************************************')
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);
disp('*******************************************************************')
disp('DO CLUSTERING STEP')
idx = kmeans(y_train,Experts,'MaxIter',500);

dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
modelNN=Classify_Clement(X_train,dd,Experts);
[dd,~] = pred_class(X_train, modelNN); % Predicts the Labels            
diff_c=max(y_train)-min(y_train);
Class_all=cell(Experts,1);
%% 

for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);


% a=cell(10,1); % You can initialise a cell this way also
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ij=1:Experts
 fprintf('Starting Expert %d... .\n', ij);     
 Classe= Class_all{ij,1}; 
 if size(Classe,1)~= 0
[net]=optimise_experts_dnn_2(X_train,y_train,Classe);    
weights_updated{ij,1}=net;

 end
 fprintf('Finished Expert %d... .\n', ij);     
end
disp('*******************************************************************')
disp('optimise classifier')
disp('*******************************************************************')

% [modelNN,updated_classtheta] = learnNN(X_train, dd, nrOfLabels,input_layer_size,...
%                hiddenLayers,layers,randInitializeWeights(layers),nnOptions );
[dd,~] = pred_class(X_train, modelNN); % Predicts the Labels              
[Valuee1,cost3]=prediction_clement_dnn_2(weights_updated,dd,X_train,...
    y_train,Class_all,Experts);
    R2ccr=cost3.R2;
    L2ccr=cost3.L2;
   RMSEccr=cost3.RMSE;
fprintf('The R2 accuracy for 1 pass CCR is %4.2f \n',R2ccr)
fprintf('The L2 accuracy for 1 pass CCR is %4.2f \n',L2ccr)
fprintf('The root mean squared error for 1 pass CCR is %4.2f \n',RMSEccr)
disp('*******************************************************************')
R2now=R2ccr; 
%% Starting MM loop
 for i=1:Inf
fprintf('iteration %d... .\n', i); 
weights=weights_updated;
if i~=1
dd = MM_clement_dnn_2(weights,X_train,y_train,modelNN,Class_all,Experts); 
end
Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ik=1:Experts
 fprintf('Starting Expert %d... .\n', ik);     
 Classe= Class_all{ik,1}; 
 if size(Classe,1)~=0
 [net]=optimise_experts_dnn_2(X_train,y_train,Classe); 
weights_updated{ik,1}=net;

 end
 fprintf('Finished Expert %d... .\n', ik);     
end
           
dd_updated = MM_clement_dnn_2(weights_updated,X_train,y_train,modelNN,Class_all,Experts);
modelNN=Classify_Clement(X_train,dd_updated,Experts);
     
 [Valuee,cost2]=prediction_clement_dnn_2(weights_updated,dd_updated,...
     X_train,y_train,Class_all,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
disp('*******************************************************************')   
R2_allmm(i,:)=double(R2);
L2_allmm(i,:)=double(L2);
RMSE_allmm(i,:)=double(RMSE);
valueallmm(:,i)=double(Valuee);
fprintf('R2 went from %4.4f to %4.4f... .\n', R2now,R2);     
if abs(R2-R2now) < (0.0001) || (i==maxitercc) || (RMSE==0.00) || (R2==100)
   break;
end
R2now=R2;
     
 end
 %%
Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd_updated==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ik=1:Experts
 fprintf('Starting Expert %d... .\n', ik);     
 Classe= Class_all{ik,1}; 
 if size(Classe,1)~=0
 [net]=optimise_experts_dnn_2(X_train,y_train,Classe); 
weights_updated{ik,1}=net;

 end
 fprintf('Finished Expert %d... .\n', ik);     
end 
modelNN=Classify_Clement(X_train,dd_updated,Experts);
 %%
oldfolder=cd;
cd(oldfolder) % setting original directory

tt=toc;
geh=[RMSEccr; RMSE_allmm];
iterr=size(geh,1);
xx=1:iterr;
figure()
subplot(2,2,1)
plot(xx,[RMSEccr; RMSE_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('RMSE') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM','location','northeast');

subplot(2,2,2)
plot(xx,[R2ccr; R2_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('R2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM','location','northeast');

subplot(2,2,3)
plot(xx,[L2ccr; L2_allmm],'r','LineWidth',1)

xlim([1 iterr])
ylabel('L2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM','location','northeast');
cd(folder)
Namefilef=strcat('performance_a', sprintf('%d',jjm),'.fig');
saveas(gcf,Namefilef)
cd(oldfolder)
%% Prediction on Training data Training accuracy);
[dd_unie,~] = pred_class(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,costhardt]=prediction_clement_dnn_2(weights_updated,...
    dd_unie,X_train,y_train,Class_all,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement_dnn_2...
    (weights_updated,modelNN,X_train,y_train,Class_all,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;
disp('*******************************************************************')

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)
[dd_unie,D] = pred_class(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test')
disp('*******************************************************************')
[Valueehard,costhard]=prediction_clement_dnn_2(weights_updated,dd_unie,...
    X_test,y_test,Class_all,Experts);
disp('predict Soft Prediction on test')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement_dnn_2...
    (weights_updated,modelNN,X_test,y_test,Class_all,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);

[hardanswer,softanswer]=Plot_perform_dnn(hardtr,softtr,hardts,softts,...
    yini,method,folder,Xini,ind_train,ind_test,oldfolder,Datause,jjm);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 
fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

cd(folder)
Namefile=strcat('Summary_', sprintf('%d',jjm),'.out');
file5 = fopen(Namefile,'w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');
Matrix=[hardanswer,softanswer];
headers = {'Hard_pred','Soft_pred'}; 
Namefile2=strcat('output_answer_', sprintf('%d',jjm),'.csv');
csvwrite_with_headers(  Namefile2,Matrix,headers);
save(strcat('R2evolution_', sprintf('%d',jjm),'.out'),...
    'R2_allmm','-ascii')
save(strcat('L2evolution_', ...
    sprintf('%d',jjm),'.out'),'L2_allmm','-ascii')
save(strcat('RMSEevolution_', sprintf('%d',jjm),'.out'),...
    'RMSE_allmm','-ascii')
save(strcat('Valueevolution_', sprintf('%d',jjm),'.out'),...
    'valueallmm','-ascii')

cd(oldfolder)

Regressors{jjm,1}=weights_updated;
Classifiers{jjm,1}=modelNN;
Classallsbig{jjm,1}= Class_all;
clfysses{jjm,1}= clfy;
clfxsses{jjm,1}= clfx;
Trainingsets{jjm,1}= [Xini,yini]; 
Expertsbig(jjm,:)=Experts;
else
disp('*******************************************************************')    
disp('-----------------------------random-MM SCHEME---------------------------') 
%  parpool('cluster1',8) 
tic;
 R2now=0; 

%% Starting MM loop
 for i=1:Inf
fprintf('iteration %d... .\n', i); 
if i==1

 dd = randi(Experts,size(y_train,1),1);
else
weights=weights_updated;
dd = MM_clement_dnn_2(weights,X_train,y_train,modelNN,Class_all,Experts); 
end

modelNN=Classify_Clement(X_train,dd,Experts);

% diff_c=max(y_train)-min(y_train);

Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
disp('*******************************************************************')
disp('Optimise experts in parallel')

parfor il=1:Experts
 fprintf('Starting Expert %d... .\n', il);   
 
 Classe= Class_all{il,1}; 
 if size(Classe,1)~= 0
 [net]=optimise_experts_dnn_2(X_train,y_train,Classe);   
weights_updated{il,1}=net;

 end
 fprintf('Finished Expert %d... .\n', il);     
end

dd_updated = MM_clement_dnn_2(weights_updated,X_train,y_train,modelNN,Class_all,Experts);

modelNN=Classify_Clement(X_train,dd_updated,Experts);
 %[dd_updated,D] = pred_class(X_train, modelNN); % Predicts the Labels 
 
 [Valuee,cost2]=prediction_clement_dnn_2(weights_updated,dd_updated,...
     X_train,y_train,Class_all,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
R2_allmm(i,:)=double(R2);
L2_allmm(i,:)=double(L2);
RMSE_allmm(i,:)=double(RMSE);
valueallmm(:,i)=double(Valuee);
fprintf('R2 went from %4.4f to %4.4f... .\n', R2now,R2);    
%if i>=2
if (abs(R2-R2now)) < (0.0001) || (i==maxitercc) || (RMSE==0.00) || (R2==100)
   break;
end
%end
R2now=R2;
    %fprintf('Finished iteration %d... .\n', i);          
 end
 %%
modelNN=Classify_Clement(X_train,dd_updated,Experts); 
%%
Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd_updated==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
disp('*******************************************************************')
disp('Optimise experts in parallel')

parfor il=1:Experts
 fprintf('Starting Expert %d... .\n', il);   
 
 Classe= Class_all{il,1}; 
 if size(Classe,1)~= 0
 [net]=optimise_experts_dnn_2(X_train,y_train,Classe);   
weights_updated{il,1}=net;

 end
 fprintf('Finished Expert %d... .\n', il);     
end
           
            %%           
oldfolder=cd;
cd(oldfolder) % setting original directory

tt=toc;
geh=[RMSE_allmm];
iterr=size(geh,1);
xx=1:iterr;
figure()
subplot(2,2,1)
plot(xx,[RMSE_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('RMSE') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('random-MM','location','northeast');

subplot(2,2,2)
plot(xx,[R2_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('R2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('random-MM','location','northeast');

subplot(2,2,3)
plot(xx,[L2_allmm],'r','LineWidth',1)

xlim([1 iterr])
ylabel('L2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('random-MM','location','northeast');
cd(folder)
Namefilef=strcat('performance_a', sprintf('%d',jjm),'.fig');
saveas(gcf,Namefilef)
cd(oldfolder)
%% Prediction on Training data Training accuracy);
[dd_unie,~] = pred_class(X_train, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on training data')
[Valueehardtr,costhardt]=prediction_clement_dnn_2(weights_updated,...
    dd_unie,X_train,y_train,Class_all,Experts);
disp('predict Soft Prediction on training data')
[Valueesoftt,costsoftt]=Unseen_soft_prediction_clement_dnn_2...
    (weights_updated,modelNN,X_train,y_train,Class_all,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
%% Prediction on Test data (Test accuracy)

[dd_unie,D] = pred_class(X_test, modelNN); % Predicts the Labels 
disp('predict Hard Prediction on test data')
[Valueehard,costhard]=prediction_clement_dnn_2(weights_updated,...
    dd_unie,X_test,y_test,Class_all,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,costsoft]=Unseen_soft_prediction_clement_dnn_2...
    (weights_updated,modelNN,X_test,y_test,Class_all,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);
[hardanswer,softanswer]=Plot_perform_dnn(hardtr,softtr,hardts,softts,...
    yini,method,folder,Xini,ind_train,ind_test,oldfolder,Datause,jjm);
fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 
fprintf('The Neural Network Classifier architecture is: [');
fprintf('%g ', layers);
fprintf(']\n');

cd(folder)
Namefile=strcat('Summary_', sprintf('%d',jjm),'.out');
file5 = fopen(Namefile,'w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
fprintf(file5,'The Neural Network Classifier architecture is: [');
fprintf(file5,'%g ', layers);
fprintf(file5,']\n');
Matrix=[hardanswer,softanswer];
headers = {'Hard_pred','Soft_pred'}; 
Namefile2=strcat('output_answer_', sprintf('%d',jjm),'.csv');
csvwrite_with_headers(  Namefile2,Matrix,headers);
save(strcat('R2evolution_', sprintf('%d',jjm),'.out'),...
    'R2_allmm','-ascii')
save(strcat('L2evolution_', ...
    sprintf('%d',jjm),'.out'),'L2_allmm','-ascii')
save(strcat('RMSEevolution_', sprintf('%d',jjm),'.out'),...
    'RMSE_allmm','-ascii')
save(strcat('Valueevolution_', sprintf('%d',jjm),'.out'),...
    'valueallmm','-ascii')
cd(oldfolder) 
Regressors{jjm,1}=weights_updated;
Classifiers{jjm,1}=modelNN;
Classallsbig{jjm,1}= Class_all;
clfysses{jjm,1}= clfy;
clfxsses{jjm,1}= clfx;
Trainingsets{jjm,1}= [Xini,yini]; 
Expertsbig(jjm,:)=Experts;
end   
end 
%end
disp('*******************************************************************')
cd(folder)
parsave2(Regressors,...
    Classifiers,Classallsbig,clfysses,clfxsses,...
Trainingsets)
save('combo.out','Ultimate_clement','-ascii')
save('Experts.out','Expertsbig','-ascii')
cd(oldfolder)
rmpath('CKS_DNN')
rmpath('Data')

    case 3
       %%
disp('*******************************************************************')
disp('BROAD OPTION OF FITTING A MODEL USING MIXTURE OF EXPERTS')
disp('*******************************************************************')
disp(' The experts are RF and the Gate is a RF')
addpath('RFS');
addpath('Data');
oldfolder=cd;

%% Select the Data to use
disp('*******************************************************************')
disp('*******************************************************************')
disp('SELECT OPTION FOR TRAINING THE MODEL')
disp('1:CCR')
disp('2:CCR-MM')
disp('3:MM-MM')
method=input('Enter the learning scheme desired: ');
if method > 3
error('Wrong choice please select 1-3')
end
if (method==2) || (method==3)
maxitercc=input('Enter the maximum iteration: '); 
%  maxitercc=20;  
end
Datause=10;
 sd=1;
 rng(sd); % set random number generator with seed sd
cd(oldfolder)
paroptions = statset('UseParallel',true);
for jjm=1:size(yb,2)
 fprintf('measurement %d | %d .\n', jjm,size(yb,2));  
 y=yb(:,jjm);
%% Summary of Data

    [a,b]=size(X);
    c=size(y,1);

%% Options for Training

disp('*******************************************************************')
%% Split dataset to Train and Test;    
Xini=X;
yini=y;
  
clfx = MinMaxScaler();
(clfx.fit(X));
X=(clfx.transform(X));

clfy = MinMaxScalery();
(clfy.fit(y));
y=(clfy.transform(y));
disp('*******************************************************************')
% Test_percentage=input('Enter the fraction of test data (in decimals) required (0.1-0.3): ');
Test_percentage=0.2;
disp('')
[X_train, X_test, y_train, y_test,ind_train,ind_test] = train_test_split...
    (X,y,Test_percentage);
%%
disp('SELECT OPTION FOR EXPERTS')
disp('1:Recommended number of Experts') % This throws an error sometimes
disp('2:User specific')

%mummy=input('Enter the options for choosing number of experts: ');
mummy=2;
if mummy > 2
error('Wrong choice please select 1-2')
end
disp('*******************************************************************')
%% Experts options
if mummy==1
        Experts=7;        
else
disp('*******************************************************************')
disp('SELECT OPTION FOR THE EXPERTS')
%Experts=input('Enter the maximum number of experts required: ');
Experts=20;

if size (X_train,1)==1
    inflate=2;
else
    inflate=10;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
%Data=y_train;
[IDX,C,SUMD,Kk]=kmeans_opt(Data,20); %Elbow method
Experts=min(Experts,Kk);
end

fprintf('The Expert that will be used is : %d \n',Experts);
disp('*******************************************************************')
%% Start Simuations for CCR,CCR-MM and MM-MM
oldfolder=cd;
cd(oldfolder) % setting original directory
if method==1
disp('*******************************************************************')    
disp('CCR SCHEME') 
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
disp('*******************************************************************')
disp('DO CLUSTERING STEP')
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);
idx = kmeans(y_train,Experts,'MaxIter',500);
dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
Mdl = TreeBagger(20,X_train,dd,'Method','c','Options',paroptions);
dd = str2double(predict(Mdl,X_train));
Class_all=cell(Experts,1);
%% 

for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);
disp('*******************************************************************')
disp('DO REGRESSION STEP')
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor i=1:Experts
 fprintf('Starting Expert %d... .\n', i);     
 Classe= Class_all{i,1}; 
 if size(Classe,1)~= 0
[net]=optimise_RF(X_train,y_train,Classe);
 weights_updated{i,1}=net;

 end
 fprintf('Finished Expert %d... .\n', i);     
end

tt=toc;
%% Prediction on Training data Training accuracy);
dd_unie = str2double(predict(Mdl,X_train));
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,stdtr,costhardt]=prediction_RF(weights_updated,dd_unie,...
    X_train,y_train,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,sstdtr,costsoftt]=Soft_prediction_RF(weights_updated,...
    Mdl,X_train,y_train,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

disp('*******************************************************************')
hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);

stdtr=clfy.inverse_transform(stdtr);
sstdtr=clfy.inverse_transform(sstdtr);
%% Prediction on Test data (Test accuracy)
dd_unie = str2double(predict(Mdl,X_test)); % Predicts the Labels 
disp('predict Hard Prediction on test data')
disp('*******************************************************************')

[Valueehard,stdte,costhard]=prediction_RF(weights_updated,dd_unie,X_test,...
    y_test,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,sstdte,costsoft]=Soft_prediction_RF(weights_updated,...
    Mdl,X_test,y_test,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);
stdte=clfy.inverse_transform(stdte);
sstdte=clfy.inverse_transform(sstdte);

[hardanswer,softanswer,ind_train,ind_test,stdclem,stdsclem]=Plot_perform...
    (hardtr,softtr,hardts,softts,yini,...
    method,folder,Xini,ind_train,ind_test,oldfolder,...
    Datause,stdtr,stdte,sstdtr,sstdte,jjm);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 

cd(folder)
Namefile=strcat('Summary_', sprintf('%d',jjm),'.out');
file5 = fopen(Namefile,'w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 


Matrix=[hardanswer,softanswer,stdclem,stdsclem];
headers = {'Hard_pred','Soft_pred','Hard_std','Soft_std'}; 
Namefile2=strcat('output_answer_', sprintf('%d',jjm),'.csv');
csvwrite_with_headers(  Namefile2,Matrix,headers);


cd(oldfolder)
Regressors{jjm,1}=weights_updated;
Classifiers{jjm,1}=Mdl;
Classallsbig{jjm,1}= Class_all;
clfysses{jjm,1}= clfy;
clfxsses{jjm,1}= clfx;
Trainingsets{jjm,1}= [Xini,yini]; 
Expertsbig(jjm,:)=Experts;
elseif method==2
R2_allmm=zeros(maxitercc,1);
L2_allmm=zeros(maxitercc,1);
RMSE_allmm=zeros(maxitercc,1);
valueallmm=zeros(size(y_train,1),maxitercc);    
    
disp('MM SCHEME')
disp('*******************************************************************')
tic;
if size (X_train,1)==1
    inflate=2;
else
    inflate=1;

end
Data=[X_train size(X_train,2)*(y_train)*inflate];
% gm = fitgmdist(Data,Experts); This can be used with Gaussian Mixture
% idx = cluster(gm,Data);
disp('*******************************************************************')
disp('DO CLUSTERING STEP')

idx = kmeans(y_train,Experts,'MaxIter',500);

dd=idx; 
disp('*******************************************************************')
disp('DO CLASSIFICATION STEP')
Mdl = TreeBagger(50,X_train,dd,'Method','c','Options',paroptions);
dd = str2double(predict(Mdl,X_train));
Class_all=cell(Experts,1);
%% 
for i=1:Experts
    Classe=find(dd==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);

% a=cell(10,1); % You can initialise a cell this way also
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor i=1:Experts
 fprintf('Starting Expert %d... .\n', i);     
 Classe= Class_all{i,1}; 
 if size(Classe,1)~= 0
[net]=optimise_RF(X_train,y_train,Classe);
 weights_updated{i,1}=net;

 end
 fprintf('Finished Expert %d... .\n', i);     
end
disp('optimise classifier')
disp('*******************************************************************')

dd= str2double(predict(Mdl,X_train)); % Predicts the Labels              
[Valuee1,std1,cost3]=prediction_RF(weights_updated,dd,X_train,y_train,...
    Experts);
    R2ccr=cost3.R2;
    L2ccr=cost3.L2;
   RMSEccr=cost3.RMSE;
fprintf('The R2 accuracy for 1 pass CCR is %4.2f \n',R2ccr)
fprintf('The L2 accuracy for 1 pass CCR is %4.2f \n',L2ccr)
fprintf('The root mean squared error for 1 pass CCR is %4.2f \n',RMSEccr)
disp('*******************************************************************')
R2now=R2ccr; 
%% Starting MM loop
 for i=1:Inf
fprintf('iteration %d... .\n', i); 
weights=weights_updated;
if i~=1
dd = MM_RF(weights,X_train,y_train,Mdl,Class_all,Experts); 
end
Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ik=1:Experts
 fprintf('Starting Expert %d... .\n', ik);     
 Classe= Class_all{ik,1}; 
 if size(Classe,1)~= 0
[net]=optimise_RF(X_train,y_train,Classe);
 weights_updated{ik,1}=net;

 end
 fprintf('Finished Expert %d... .\n', ik);     
end
           
dd_updated = MM_RF(weights_updated,X_train,y_train,Mdl,Class_all,Experts);
Mdl = TreeBagger(50,X_train,dd_updated,'Method','c','Options',paroptions);              
 [Valuee,~,cost2]=prediction_RF(weights_updated,dd_updated,X_train,...
     y_train,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
disp('*******************************************************************')   
R2_allmm(i,:)=double(R2);
L2_allmm(i,:)=double(L2);
RMSE_allmm(i,:)=double(RMSE);
valueallmm(:,i)=double(Valuee);
fprintf('R2 went from %4.4f to %4.4f... .\n', R2now,R2);    
if abs(R2-R2now) < (0.0001) || (i==maxitercc) || (RMSE==0.00) || (R2==100)
   break;
end
R2now=R2;     
 end
 %%
Class_all=cell(Experts,1);
%% 

for i=1:Experts
    Classe=find(dd_updated==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);
% a=cell(10,1); % You can initialise a cell this way also
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ij=1:Experts
 fprintf('Starting Expert %d... .\n', ij);     
 Classe= Class_all{ij,1}; 
 if size(Classe,1)~= 0
[net]=optimise_RF(X_train,y_train,Classe);
 weights_updated{ij,1}=net;

 end
 fprintf('Finished Expert %d... .\n', ij);    
end 
 %%
oldfolder=cd;
cd(oldfolder) % setting original directory

tt=toc;
geh=[RMSEccr; RMSE_allmm];
iterr=size(geh,1);
xx=1:iterr;
figure()
subplot(2,2,1)
plot(xx,[RMSEccr; RMSE_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('RMSE') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM','location','northeast');

subplot(2,2,2)
plot(xx,[R2ccr; R2_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('R2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM','location','northeast');

subplot(2,2,3)
plot(xx,[L2ccr; L2_allmm],'r','LineWidth',1)

xlim([1 iterr])
ylabel('L2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('MM','location','northeast');
cd(folder)
Namefilef=strcat('performance_a', sprintf('%d',jjm),'.fig');
saveas(gcf,Namefilef)
cd(oldfolder)
%% Prediction on Training data Training accuracy);
dd_unie= str2double(predict(Mdl,X_train));
disp('predict Hard Prediction on training data')
disp('*******************************************************************')
[Valueehardtr,stdtr,costhardt]=prediction_RF(weights_updated,dd_unie,...
    X_train,y_train,Experts);
disp('predict Soft Prediction on training data')
disp('*******************************************************************')
[Valueesoftt,sstdtr,costsoftt]=Soft_prediction_RF(weights_updated,...
    Mdl,X_train,y_train,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;
disp('*******************************************************************')

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
stdtr=clfy.inverse_transform(stdtr);
sstdtr=clfy.inverse_transform(sstdtr);
%% Prediction on Test data (Test accuracy)

dd_unie= str2double(predict(Mdl,X_test));
disp('predict Hard Prediction on test')
disp('*******************************************************************')
[Valueehard,stdte,costhard]=prediction_RF(weights_updated,dd_unie,X_test,...
    y_test,Experts);
disp('predict Soft Prediction on test')
disp('*******************************************************************')
[Valueesoft,sstdte,costsoft]=Soft_prediction_RF(weights_updated,...
    Mdl,X_test,y_test,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp('*******************************************************************')
disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);
stdte=clfy.inverse_transform(stdte);
sstdte=clfy.inverse_transform(sstdte);
[hardanswer,softanswer,ind_train,ind_test,stdclem,stdsclem]=Plot_perform...
    (hardtr,softtr,hardts,softts,yini,...
    method,folder,Xini,ind_train,ind_test,oldfolder,Datause,stdtr,stdte,...
    sstdtr,sstdte,jjm);

fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 


cd(folder)
Namefile=strcat('Summary_', sprintf('%d',jjm),'.out');
file5 = fopen(Namefile,'w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 

save(strcat('R2evolution_', sprintf('%d',jjm),'.out'),...
    'R2_allmm','-ascii')
save(strcat('L2evolution_', ...
    sprintf('%d',jjm),'.out'),'L2_allmm','-ascii')
save(strcat('RMSEevolution_', sprintf('%d',jjm),'.out'),...
    'RMSE_allmm','-ascii')
save(strcat('Valueevolution_', sprintf('%d',jjm),'.out'),...
    'valueallmm','-ascii')

Matrix=[hardanswer,softanswer,stdclem,stdsclem];
headers = {'Hard_pred','Soft_pred','Hard_std','Soft_std'}; 
Namefile2=strcat('output_answer_', sprintf('%d',jjm),'.csv');
csvwrite_with_headers(  Namefile2,Matrix,headers);

cd(oldfolder)
Regressors{jjm,1}=weights_updated;
Classifiers{jjm,1}=Mdl;
Classallsbig{jjm,1}= Class_all;
clfysses{jjm,1}= clfy;
clfxsses{jjm,1}= clfx;
Trainingsets{jjm,1}= [Xini,yini]; 
Expertsbig(jjm,:)=Experts;
else
disp('*******************************************************************')    
  disp('random-MM SCHEME') 
%  parpool('cluster1',8) 
R2_allmm=zeros(maxitercc,1);
L2_allmm=zeros(maxitercc,1);
RMSE_allmm=zeros(maxitercc,1);
valueallmm=zeros(size(y_train,1),maxitercc);    
tic;
 R2now=0; 
%% Starting MM loop
 for i=1:Inf
fprintf('iteration %d... .\n', i); 
if i==1

 dd = randi(Experts,size(y_train,1),1);
 disp('Initialised randomly for the first time')
else
weights=weights_updated;
dd = MM_RF(weights,X_train,y_train,Mdl,Class_all,Experts);
disp('initialised using MM scheme')
end

Class_all=cell(Experts,1);
for ii=1:Experts
Classe=find(dd==ii);
Class_all{ii,1}=Classe;    
end 

weights_updated=cell(Experts,1);
disp('*******************************************************************')
disp('Optimise experts in parallel')

parfor il=1:Experts
 fprintf('Starting Expert %d... .\n', il);     
 Classe= Class_all{il,1}; 
 if size(Classe,1)~= 0
[net]=optimise_RF(X_train,y_train,Classe);
 weights_updated{il,1}=net;

 end
 fprintf('Finished Expert %d... .\n', il);    
end

if i==1
[Valueeini,~,costini]=prediction_RF(weights_updated,dd,X_train,y_train,...
    Experts);
fprintf('R2 initial accuracy for random initialisation is %4.4f... .\n', costini.R2);   
end

if i==1
dd_updated=dd;
else
dd_updated = MM_RF(weights_updated,X_train,y_train,Mdl,Class_all,Experts);
end

Mdl = TreeBagger(20,X_train,dd_updated,'Method','c','Options',paroptions);   
 [Valuee,~,cost2]=prediction_RF(weights_updated,dd_updated,X_train,...
     y_train,Experts);
    R2=cost2.R2;
    L2=cost2.L2;
   RMSE=cost2.RMSE;
R2_allmm(i,:)=double(R2);
L2_allmm(i,:)=double(L2);
RMSE_allmm(i,:)=double(RMSE);
valueallmm(:,i)=double(Valuee);
fprintf('R2 went from %4.4f to %4.4f... .\n', R2now,R2);    
%if i>=2
if (abs(R2-R2now)) < (0.0001) || (i==maxitercc) || (RMSE==0.00) || (R2==100)
   break;
end
%end
R2now=R2;
    fprintf('Finished iteration %d... .\n', i);          
 end
 %%
 Class_all=cell(Experts,1);
%% 
for i=1:Experts
    Classe=find(dd_updated==i);
    Class_all{i,1}=Classe;
    
end 
weights_updated=cell(Experts,1);
% a=cell(10,1); % You can initialise a cell this way also
disp('*******************************************************************')
disp('Optimise experts in parallel')
parfor ij=1:Experts
 fprintf('Starting Expert %d... .\n', ij);     
 Classe= Class_all{ij,1}; 
 if size(Classe,1)~= 0
[net]=optimise_RF(X_train,y_train,Classe);
 weights_updated{ij,1}=net;

 end
 fprintf('Finished Expert %d... .\n', ij);    
end

%%           
oldfolder=cd;
cd(oldfolder) % setting original directory
%folder = strcat('Results__MM_MM', sprintf('%.3d',jjm));

tt=toc;
geh=[RMSE_allmm];
iterr=size(geh,1);
xx=1:iterr;
figure()
subplot(2,2,1)
plot(xx,[RMSE_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('RMSE') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('random-MM','location','northeast');

subplot(2,2,2)
plot(xx,[R2_allmm],'r','LineWidth',1)
xlim([1 iterr])
ylabel('R2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('random-MM','location','northeast');

subplot(2,2,3)
plot(xx,[L2_allmm],'r','LineWidth',1)

xlim([1 iterr])
ylabel('L2 accuracy in %') 
xlabel('iterations') 
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
legend('random-MM','location','northeast');
cd(folder)
Namefilef=strcat('performance_a', sprintf('%d',jjm),'.fig');
saveas(gcf,Namefilef)
cd(oldfolder)
%% Prediction on Training data Training accuracy);

dd_unie= str2double(predict(Mdl,X_train));
disp('predict Hard Prediction on training data')
[Valueehardtr,stdtr,costhardt]=prediction_RF(weights_updated,dd_unie,...
    X_train,y_train,Experts);
disp('predict Soft Prediction on training data')
[Valueesoftt,sstdtr,costsoftt]=Soft_prediction_RF(weights_updated,...
    Mdl,X_train,y_train,Experts);
R2hardt=costhardt.R2;
R2softt=costsoftt.R2;

hardtr=clfy.inverse_transform(Valueehardtr);
softtr=clfy.inverse_transform(Valueesoftt);
stdtr=clfy.inverse_transform(stdtr);
sstdtr=clfy.inverse_transform(sstdtr);
%% Prediction on Test data (Test accuracy)
%[dd_unie,D] = pred_class(X_test, modelNN); % Predicts the Labels 
[dd_unie]= str2double(predict(Mdl,X_test));
disp('predict Hard Prediction on test data')
[Valueehard,stdte,costhard]=prediction_RF(weights_updated,dd_unie,X_test,...
    y_test,Experts);
disp('predict Soft Prediction on test data')
disp('*******************************************************************')
[Valueesoft,sstdte,costsoft]=Soft_prediction_RF(weights_updated,...
    Mdl,X_test,y_test,Experts);
R2hard=costhard.R2;
R2soft=costsoft.R2;

disp(' Rescale back the predictions and save to file')
hardts=clfy.inverse_transform(Valueehard);
softts=clfy.inverse_transform(Valueesoft);
stdte=clfy.inverse_transform(stdte);
sstdte=clfy.inverse_transform(sstdte);
[hardanswer,softanswer,ind_train,ind_test,stdclem,stdsclem]=Plot_perform...
    (hardtr,softtr,hardts,softts,yini,...
    method,folder,Xini,ind_train,ind_test,oldfolder,Datause,stdtr,stdte,...
    sstdtr,sstdte,jjm);
fprintf('The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf('The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf('The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf('The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf('The wall clock time is %4.2f secs \n',tt);  
fprintf('The number of experts used is %d \n',Experts); 

cd(folder)
Namefile=strcat('Summary_', sprintf('%d',jjm),'.out');
file5 = fopen(Namefile,'w+'); 
fprintf(file5,'The R2 accuracy for hard prediction on (training data) is %4.2f \n',R2hardt); 
fprintf(file5,'The R2 accuracy for soft prediction on (training data) is %4.2f \n',R2softt); 
fprintf(file5,'The R2 accuracy for hard prediction on (test data) is %4.2f \n',R2hard); 
fprintf(file5,'The R2 accuracy for soft prediction on (test data) is %4.2f \n',R2soft); 
fprintf(file5,'The wall clock time is %4.2f secs \n',tt); 
fprintf(file5,'The number of experts used is %d \n',Experts); 
save(strcat('R2evolution_', sprintf('%d',jjm),'.out'),...
    'R2_allmm','-ascii')
save(strcat('L2evolution_', ...
    sprintf('%d',jjm),'.out'),'L2_allmm','-ascii')
save(strcat('RMSEevolution_', sprintf('%d',jjm),'.out'),...
    'RMSE_allmm','-ascii')
save(strcat('Valueevolution_', sprintf('%d',jjm),'.out'),...
    'valueallmm','-ascii')
Matrix=[hardanswer,softanswer,stdclem,stdsclem];
headers = {'Hard_pred','Soft_pred','Hard_std','Soft_std'}; 
Namefile2=strcat('output_answer_', sprintf('%d',jjm),'.csv');
csvwrite_with_headers(  Namefile2,Matrix,headers);


cd(oldfolder)  

Regressors{jjm,1}=weights_updated;
Classifiers{jjm,1}=Mdl;
Classallsbig{jjm,1}= Class_all;
clfysses{jjm,1}= clfy;
clfxsses{jjm,1}= clfx;
Trainingsets{jjm,1}= [Xini,yini]; 
Expertsbig(jjm,:)=Experts;
end 
disp('*******************************************************************')

end
cd(folder)
parsave2(Regressors,...
    Classifiers,Classallsbig,clfysses,clfxsses,...
Trainingsets)
save('combo.out','Ultimate_clement','-ascii')
save('Experts.out','Expertsbig','-ascii')
cd(oldfolder)

rmpath('RFS')
rmpath('data')
end
disp('*******************PROGRAMME EXECUTED******************************')