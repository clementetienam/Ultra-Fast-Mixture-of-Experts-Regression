function [Valuee,variance2]=Unseen_soft_3(weights,modelNN,X,Experts,clfy)

      [~,D] = pred_class(X,modelNN);


numcols=size(D,2);
Valuee=zeros(size(X,1),1);
variance2=zeros(size(X,1),numcols);
%%
parfor jj=1:size(X,1)
    
    a00=X(jj,:) ; 
    Valuee1=zeros(1,numcols);
    Valuees1=zeros(1,numcols);
    for xx= 1:Experts
        
    net=weights{xx,:};
    [zz,s2] = predict(net,a00);
    zz=reshape(zz,[],1);
    s2=reshape(s2,[],1);
    Valuee1(:,xx)= zz;
    Valuees1(:,xx)= s2; 
        end

getit=D(jj,:).*Valuee1;
Valuee(jj,:)=clfy.inverse_transform(sum(getit,2));
variance2(jj,:)=sqrt(clfy.inverse_transform(sum((D(jj,:).*Valuees1),2)+...
    (sum((D(jj,:).*(Valuee1.^2)),2)-...
    (sum((D(jj,:).*Valuee1),2)).^2)));
end
end