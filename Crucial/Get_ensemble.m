function ensemble_ini=Get_ensemble(N,X_test2,meanss2,meanss)
%  for j=1:N
%     for jj=1:size(X_test2,2)
%     hyp_inipure(:,jj) = normrnd( meanss2(:,jj),0.1*meanss2(:,jj),1,1) ; 
%     end 
%     hyp_inipure=abs(hyp_inipure);
%     hyp_inipure(:,6)=round(hyp_inipure(:,6));
%     hyp_inipure(:,8)=round(hyp_inipure(:,8));
%     hyp_ini=(clfx.transform(hyp_inipure));   
%     ensemble_ini(:,j)=hyp_ini';
%  end
 
 for jj=1:size(X_test2,2)
  hyp_inipuree(1:N,jj) = unifrnd(meanss2(:,jj),meanss(:,jj),10,1);     
 end
ensemble_ini=hyp_inipuree';
end