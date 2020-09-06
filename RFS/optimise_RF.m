function [hyper_updated]=optimise_RF(X_train,y_train,Classe)
Xuse=X_train(Classe,:);
yuse=y_train(Classe,:);
if size(Xuse,1)~= 0

hyper_updated = TreeBagger(500,Xuse,yuse,'Method','regression');
end

end