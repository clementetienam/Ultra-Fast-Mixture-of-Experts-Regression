function ensemble_ini=Get_ensemble_2(N,X_test2,meanss2,meanss,Nop)

p=2;
szss=Nop;
for k=1:N
 for jj=1:size(X_test2,2)
 aj=meanss2(:,jj)+ (meanss(:,jj)- meanss2(:,jj))*sum(rand(szss,p),2)/p;
 % hyp_inipuree(:,jj) = unifrnd(meanss2(:,jj),meanss(:,jj),szss,1);    
   hyp_inipuree(:,jj) = reshape(aj,[],1);    
end

ensemble_ini(:,k)=reshape(hyp_inipuree,[],1);
end