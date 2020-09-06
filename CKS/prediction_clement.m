function [Valuee,Valuees,cost]=prediction_clement(weights,dd_updated,X,y,Xtrains,ytrains,Experts)
  labelDA =dd_updated; %target prediction
  % Here we should first prediction the class labels, i.e:
  % [~,D] = predictNN(X, modelNN); 
  % for i=1:size(X,1)
  %     [clem,clem2]=max(D(i,:));
  %     labelDA(i)=clem2;
  % end

	meanfunc=@meanConst;
likfunc = {@likGauss};    

inf = @infGaussLik;
 cov = {@covSEiso}; 
 infv  = @(varargin) inf(varargin{:},struct('s',1.0));   
for i=1:Experts

    indee=find(labelDA==i);
   
    hyp_use=weights{i,:};
    Xuse=Xtrains{i,:};
    yuse=ytrains{i,:};
     if size(Xuse,1)>= 2
	cov1 = {'apxSparse',cov,hyp_use.xu};  
    
    
         
    a00=X(indee,:) ; 


    [zz s2] = gp(hyp_use, infv, meanfunc, cov1, likfunc, Xuse, yuse, a00);

zz=reshape(zz,[],1);
s2=reshape(s2,[],1);

Valuee(indee,:)= zz;
Valuees(indee,:)= s2;
     else

       Valuee(indee,:)= 0; % This should be mean function 
       % Also  Valuees(indee,:)=  should be magnitude + likelihood variance
     end
        

	end



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