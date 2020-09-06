function [Valuee]=prediction_2(weights,dd_updated,X,Class_all,Experts,clfy)
  labelDA =dd_updated; %target prediction
%%
parfor jj=1:size(labelDA,1)
    label=labelDA(jj,:);
    net=weights{label,:};
    a00=X(jj,:) ; 
    zz = (predict(net,a00'))';
    zz=reshape(zz,[],1);

    Valuee(jj,:)= zz;
end
Valuee=clfy.inverse_transform(Valuee);

end