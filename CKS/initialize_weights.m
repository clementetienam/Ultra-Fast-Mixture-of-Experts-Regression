function weights=initialize_weights(diff_c)
% if methss==1
    hyp=log([5;(diff_c/10)]); % 10 gave the best for now
    hyp2=log(diff_c/10);
    hyp3=[hyp;hyp2];
% else
    
    
%      hyp=0.2*randn(2,1);
%      hyp=log([10;9]);
%     %hyp=0.2*normrnd(0,1,2,1);
%      %hyp2=0.2*randn(1,1);
%     %hyp2=0.2*normrnd(0,1,1,1);
%     hyp2=20;
%     hyp3=[hyp;hyp2];
% end

weights=hyp3;
end