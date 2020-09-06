function labels = MM_RF(weights,X,y,modelNN,Class_all,Experts)

 dd=size(X,1);

 outputtR=zeros(dd,Experts);

for L=1:Experts
    Classuse=Class_all{L,:};
    if size(X(Classuse),1)~= 0   
    net=weights{L,:};
   
[m,~] = predict(net,X) ;   

m=reshape(m,[],1);
  
     outputtR(:,L)=m;

    else
    outputtR(:,L)=zeros(size(X,1),1);

    end


end

%% softmax
for i=1:Experts

thirds_term(:,i)= ((y-outputtR(:,i)).^2);

end

for i=1:size(X,1)
[~,clem2]=min(thirds_term(i,:));
clemall(:,i)=clem2;
end
labels=clemall';
end