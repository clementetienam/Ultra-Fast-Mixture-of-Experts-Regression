function out=Optimize_clement_Gp(parameters,fv,clfx,y_train,...
    oldfolder,ytrue,combo1,Regressors,Classifiers,...
   Xtrainbig,ytrainbig,Experts,clfysses)


parameters=reshape(parameters,[],8);
X_test=(clfx.transform(parameters));
cd(oldfolder);

%%
for ii=1:2

switch combo1
    case 1
[Hardmean(:,ii)]=prediction_1(Regressors{ii,1},...
    pred_class(X_test, Classifiers{ii,1})...
,X_test,Xtrainbig{ii,1},ytrainbig{ii,1},Experts,clfysses{ii,1});
% [Softmean(:,ii),~]=Unseen_soft_1(Regressors{ii,1},...
%     Classifiers{ii,1},X_test,Xtrainbig{ii,1},ytrainbig{ii,1}...
%,Experts,clfy);

    case 2

[Hardmean(:,ii)]=prediction_2(Regressors{ii,1},pred_class(X_test...
    , Classifiers{ii,1})...
,X_test,Classallsbig{ii,1},Experts,clfysses{ii,1});

% [Softmean(:,ii)]=Unseen_soft_2(Regressors{ii,1},...
%     Classifiers{ii,1},X_test,Classallsbig{ii,1},Experts,clfysses{ii,1});
    case 3

cd(folderk)
Classifiers=load('Classifiers'); 
Classifiers=Classifiers.Classifiers;

Experts=load('Experts.out');

Classallsbig=load('Classallsbig');
Classallsbig=Classallsbig.Classallsbig;


clfysses=load('clfysses'); 
clfysses=clfysses.clfysses;

Regressors=load('Regressors');
Regressors=Regressors.Regressors;  

% cd(oldfolder)
[Hardmean(:,ii)]=prediction_3(Regressors{ii,1}...
    ,str2double(predict(Classifiers{ii,1},X_test))...
,X_test,Classallsbig{ii,1},Experts,clfysses{ii,1});

% [Softmean(:,ii),~]=Unseen_soft_3(Regressors{ii,1}...
%     ,Classifiers{ii,1},X_test,Experts,clfysses{ii,1});    
%     
end
 cd(oldfolder)
end
% fprintf('Done measurement %d | %d .\n', ii,2);   

%end

%%

%%

ytrue=reshape(ytrue,[],1);
Hardmean=double(reshape(Hardmean,[],1));
gg=size(ytrue,1);
% a1=((ytrue'-Hardmean').^2).^0.5;
a1=(1/(2*gg)) * sum((ytrue-Hardmean).^2);

%a1=abs((ytrue-Hardmean));

cc = sum(a1);
out=cc;

end