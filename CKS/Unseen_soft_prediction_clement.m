function [Valuee,variance2,cost]=Unseen_soft_prediction_clement(weights,modelNN,X,y,Xtrains,ytrains,Experts)

   %[~,D] = predictNN(X, modelNN); 
[~,D]=pred_class(X,modelNN);
a = sum(D,2);
D = bsxfun(@rdivide, D, a);   

numcols=size(D,2);
Valueer=zeros(size(X,1),numcols);
Term_1=zeros(size(X,1),numcols);
Term_2=zeros(size(X,1),numcols);
Term_3=zeros(size(X,1),numcols);
	meanfunc=@meanConst;
likfunc = {@likGauss};    

inf = @infGaussLik;
 cov = {@covSEiso}; 
 infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
for i=1:Experts
    
    hyp_use=weights{i,:};
    Xuse=Xtrains{i,:};
    if size(Xuse,1)~= 0
    yuse=ytrains{i,:};
	cov1 = {'apxSparse',cov,hyp_use.xu};  
    a00=X ; 
    [zz, s2] = gp(hyp_use, infv, meanfunc, cov1, likfunc, Xuse, yuse, a00);

zz=reshape(zz,[],1);
    else
        zz=0;
        s2=0;
    end
getit=D(:,i).*zz;
Valueer(:,i)=getit;
Term_1(:,i)=D(:,i).*s2;
Term_2(:,i)=D(:,i).*(zz.^2);
Term_3(:,i)=D(:,i).*zz;
end

Valuee=sum(Valueer,2);
variance2=sum(Term_1,2)+(sum(Term_2,2)-(sum(Term_3,2)).^2);

   CCR=Valuee;
   True=y;
   yesup=sum((abs(CCR-True)).^2);
   yesdown=sum((True-mean(True,1)).^2);
   R2=1-(yesup/yesdown);
   R2=R2*100;
   yesdown2=sum((abs(True)).^2);
   L2=1-((yesup/yesdown2).^0.5);
   L2=L2*100;
   matrix=True-CCR;
   L1norm=norm(matrix,1);
   L2norm=norm(matrix,2);
   absolute_error=abs(matrix);
   MAE=mean(absolute_error,1);
   squared_error=(absolute_error).^2;
   MSE=mean(squared_error);
   RMSE=MSE.^0.5;
  
   cost.R2=R2;
   cost.L2=L2;
   cost.L1norm=L1norm;
   cost.L2norm=L2norm;
   cost.MAE=MAE;
   cost.RMSE=RMSE;
end