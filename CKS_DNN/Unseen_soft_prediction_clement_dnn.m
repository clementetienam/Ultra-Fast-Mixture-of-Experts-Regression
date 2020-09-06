function [Valuee,cost]=Unseen_soft_prediction_clement_dnn(weights,modelNN,X,y,Class_all,Experts)

   [labelDA,D] = predictNN(X, modelNN); 


Valueer=zeros(size(X,1),Experts);

for i=1:Experts
    
    net=weights{i,:};

 Classe= Class_all{i,1}; 
 if size(Classe,1)~= 0


         
    a00=X ; 
	zz=(mlpfwd(net, a00));



zz=reshape(zz,[],1);
    else
        zz=0;
    end
getit=D(:,i).*zz;
Valueer(:,i)=getit;

end

Valuee=sum(Valueer,2);

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