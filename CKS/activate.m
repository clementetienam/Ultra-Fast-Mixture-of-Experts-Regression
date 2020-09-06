function y = activate(x,W,b)
%ACTIVATE Evaluates ReLU function.
%
% x is the input vector, y is the output vector
% W contains the weights, b contains the shifts
%
% The ith component of y is activate((Wx+b)_i)
% where activate(z) = 1/(1+exp(-z))
% y = 1./(1+exp(-(W*x+b)));
yy=W*x+b;
yy(yy<0)=0;
y=yy;