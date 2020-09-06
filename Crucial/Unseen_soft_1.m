function [Valuee,variance2]=Unseen_soft_1(weights,...
    modelNN,X,Xtrains,ytrains,Experts,clfy)

   [~,D] = pred_class(X, modelNN); 

numcols=size(D,2);
Valuee=zeros(size(X,1),1);
variance2=zeros(size(X,1),numcols);
meanfunc=@meanConst;
likfunc = {@likGauss};    

inf = @infGaussLik;
 cov = {@covSEiso}; 
 infv  = @(varargin) inf(varargin{:},struct('s',1.0));   

%%

parfor jj=1:size(X,1)
    
    a00=X(jj,:) ; 
    Valuee1=zeros(1,numcols);
    Valuees1=zeros(1,numcols);
    for xx= 1:Experts
        
    hyp_use=weights{xx,:};
    Xuse=Xtrains{xx,:};
    yuse=ytrains{xx,:};
    cov1 = {'apxSparse',cov,hyp_use.xu};  

    [zz ,s2] = gp(hyp_use, infv, meanfunc, cov1, likfunc, Xuse, yuse, a00);

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