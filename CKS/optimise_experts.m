function [hyper_updated,Xuse,yuse]=optimise_experts(diff_c,X_train,y_train,Classe,meanfunc,likfunc,inf,cov,infv,method2)
hyper_in=initialize_weights(diff_c);
Xuse=X_train(Classe,:);
yuse=y_train(Classe,:);
if size(Xuse,1)~= 0
hyper_out.mean=0;    
hyper_out.cov=hyper_in(1:2,:);
hyper_out.lik=hyper_in(3,:);
xsparse=get_inducing_kmeans(Xuse,method2);
hyper_out.xu=xsparse;
cov1 = {'apxSparse',cov,xsparse};  
hyper_updated = minimize(hyper_out, @gp, -500, infv, meanfunc, cov1, likfunc, Xuse,yuse);

end
%     hyper_in=initialize_weights(diff_c);
%     hyper_out.cov=hyper_in(1:2,:);
%     hyper_out.lik=hyper_in(3,:);
%     xsparse=get_inducing_kmeans(Xuse,method2);
%     hyper_out.xu=xsparse;
%      hyper_updated=[];
%      
% end
end