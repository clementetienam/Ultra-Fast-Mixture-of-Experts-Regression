function cov_big= Get_cov(X_test2,hyp_updated)
for ik=1:size(X_test2,2)
cov_big(:,ik)=cov(hyp_updated(:,ik));
end