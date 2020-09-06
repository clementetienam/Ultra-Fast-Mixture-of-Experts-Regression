function [labelss,YPred]=pred_class(X,hyper_updated)
A=X;
YPred = (predict(hyper_updated,A'))';
for i=1:size(A,1)
[clem,clem2]=max(YPred(i,:));
clemall(i,:)=clem2;
end
labelss=clemall;
end