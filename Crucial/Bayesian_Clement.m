function DupdateK=Bayesian_Clement(hypps,N,f,clfx,ytrain,oldfolder,ytrue,alpha,combo1)
% ytrain=y_train;
Sim=zeros(2,N);
for i=1:N
    aa=(hypps(:,i))';
Sim(:,i)=abs((Forwarding(aa,f,clfx,ytrain,oldfolder,combo1))');
end

% ytrue=y_train(5,:);

[DupdateK] = ESMDA (hypps,ytrue', N, Sim,alpha);
end