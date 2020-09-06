function DupdateK=Bayesian_Clement_2(hypps,N,f,clfx,ytrain,...
    oldfolder,ytrue,alpha,combo1,suni)
% ytrain=y_train;
%Sim=zeros(2,N);
parfor i=1:N
    aa=(hypps(:,i));
	aa=reshape(aa,[],suni)
	spit=abs((Forwarding(aa,f,clfx,ytrain,oldfolder,combo1)));
	spit=reshape(spit,[],1);
Sim(:,i)=spit;
end

% ytrue=y_train(5,:);

[DupdateK] = ESMDA (hypps,reshape(ytrue,[],1), N, Sim,alpha);
end