function [Valuee]=prediction_1(weights,...
dd_updated,X,Xtrains,ytrains,Experts,clfys)
  labelDA =dd_updated; %target prediction

	meanfunc=@meanConst;
likfunc = {@likGauss};    

inf = @infGaussLik;
 cov = {@covSEiso}; 
 infv  = @(varargin) inf(varargin{:},struct('s',1.0));   


parfor jj=1:size(labelDA,1)
    label=labelDA(jj,:);
    hyp_use=weights{label,:};
    Xuse=Xtrains{label,:};
    yuse=ytrains{label,:};
    cov1 = {'apxSparse',cov,hyp_use.xu};  
    a00=X(jj,:) ; 
    [zz s2] = gp(hyp_use, infv, meanfunc, cov1, likfunc, Xuse, yuse, a00);

    zz=reshape(zz,[],1);
    s2=reshape(s2,[],1);

    Valuee(jj,:)= zz;
    Valuees(jj,:)= s2;
end
Valuee=clfys.inverse_transform(Valuee);
Valuees=clfys.inverse_transform(Valuees);
Valuees=sqrt(Valuees);

end