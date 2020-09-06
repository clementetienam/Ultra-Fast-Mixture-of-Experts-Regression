function [Valuee,cost]=prediction_clement_dnn(weights,dd_updated,X,y,Class_all,Experts)
  labelDA =dd_updated; %target prediction

for i=1:Experts

    indee=find(labelDA==i);
   
    net=weights{i,:};

 Classe= Class_all{i,1}; 
 if size(Classe,1)~= 0

    
    
         
    a00=X(indee,:) ; 
	

    zz=(mlpfwd(net, a00));


zz=reshape(zz,[],1);
Valuee(indee,:)= zz;

     else

       Valuee(indee,:)= 0; 
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