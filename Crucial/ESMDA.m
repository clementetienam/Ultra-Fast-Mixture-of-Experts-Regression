function [DupdateK] = ESMDA (sgsim,f, N, Sim1,alpha)
sd=1;
rng(sd)

%-----------------------------------------------------------------------------    

parfor i=1:size(f,1)
       aa=f(i,:);
    aa1=num2str(aa);
    usee=str2num(aa1(1));    
   stddWOPR1 =1 %0.01*f(i,:);
   stdall(i,:)=stddWOPR1;
   end

nobs = length(f);
noise = randn(max(10000,nobs),1);

Error1=stdall;
sig=Error1;
parfor i = 1 : length(f)
           f(i) = f(i) + sig(i)*noise(end-nobs+i);
end
R = sig.^2;
  Dj = repmat(f, 1, N);
           parfor i = 1:size(Dj,1)
             rndm(i,:) = randn(1,N); 
             rndm(i,:) = rndm(i,:) - mean(rndm(i,:)); 
             rndm(i,:) = rndm(i,:) / std(rndm(i,:));
             Dj(i,:) = Dj(i,:) + sqrt(alpha)*sqrt(R(i)) * rndm(i,:);
           end


Cd2 =diag(R);
overall=sgsim;

Y=overall; %State variable,it is important we include simulated measurements in the ensemble state variable
M = mean(Sim1,2);
% Mean of the ensemble state
M2=mean(overall,2);
%M=M'
% Get the ensemble states pertubations
parfor j=1:N
    S(:,j)=Sim1(:,j)-M;
end
parfor j=1:N
    yprime(:,j)=overall(:,j)-M2;
end
Cyd=(yprime*S')./((N-1));
Cdd=(S*S')./((N-1));

Ynew=Y+(Cyd*pinv2((Cdd+(alpha.*Cd2))))*(Dj-Sim1);

DupdateK=Ynew;

end