function inducing_points=get_inducing_kmeans(X,method)
if size(X,1)>= 2
if method==1
numrowss=round(0.3*(size(X,1)));
numrows=min(numrowss,100);
%numrows=numrowss;
numcols=size(X,2);
	for j=1:numcols

      [idx,C] = kmeans(X(:,j),numrows,'MaxIter',500);
      inducing_points(:,j)=C;

	end
else
numrowss=round(0.3*(size(X,1)));
%numrows=numrowss;
numrows=min(numrowss,100);
numcols=size(X,2);
	for j=1:numcols
        xu = normrnd(0,1,numrows,1); 
        inducing_points(:,j)=xu;
    end
end
end
end