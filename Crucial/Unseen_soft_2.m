function [Valuee]=Unseen_soft_2(weights,modelNN,X,Class_all,Experts,clfy)
   [~,D] = pred_class(X, modelNN); 


Valuee=zeros(size(X,1),1);
%%
parfor jj=1:size(X,1)
    
    a00=X(jj,:) ; 
    Valuee1=zeros(1,numcols);
    for xx= 1:Experts
     net=weights{xx,:};
     zz = (predict(net,a00'))';
     zz=reshape(zz,[],1);
     Valuee1(:,xx)= zz;
    end
Valuee(jj,:)=clfy.inverse_transform(sum((D(jj,:).*Valuee1),2));   
end

end