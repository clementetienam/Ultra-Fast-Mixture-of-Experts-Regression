function [Valuee,Valuees]=prediction_3(weights,...
    dd_updated,X,Class_all,Experts,clfy)
  labelDA =dd_updated; %target prediction

% for i=1:Experts
% 
%     indee=find(labelDA==i);
%  if size(indee,1)~= 0 
%      
%    
%     net=weights{i,:};
%  
%     a00=X(indee,:) ;     
%     [zz,stdclm] = predict(net,a00) ;   
% 
% zz=reshape(zz,[],1);
% Valuee(indee,:)= zz;
% Valuees(indee,:)=stdclm;
% 
%      else
% 
%        Valuee(indee,:)= 0; 
%        Valuees(indee,:)= 0; 
%      end
%         
% 
% 	end
% Valuee=double(Valuee);
% 
% Valuee=clfy.inverse_transform(Valuee);

%%
parfor jj=1:size(labelDA,1)
    label=labelDA(jj,:);
    net=weights{label,:};
    a00=X(jj,:) ; 
 
    [zz,s2] = predict(net,a00) ;   
    zz=reshape(zz,[],1);
    s2=reshape(s2,[],1);

    Valuee(jj,:)= double(zz);
    Valuees(jj,:)= double(s2);
end
Valuee=clfy.inverse_transform(Valuee);
Valuees=clfy.inverse_transform(Valuees);
Valuees=sqrt(Valuees);

end