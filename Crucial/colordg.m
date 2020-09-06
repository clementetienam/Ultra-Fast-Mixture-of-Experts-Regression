function linecolor = colordg(n);
%COLORDG - Choose 1 out of 10 different colors for a line plot
%The first seven colors are exactly the same as Matlab's default
%AXES Colororder property.
%
%Syntax: linecolor = colordg(n);
%
%Input: N , value between 1 and 10, giving the following colors
%
% 1 BLUE
% 2 GREEN (medium dark)
% 3 RED
% 4 TURQUOISE
% 5 MAGENTA
% 6 YELLOW (dark)
% 7 GREY (very dark)
% 8 ORANGE
% 9 BROWN
% 10 YELLOW (pale)
%
%Output: LINECOLOR (1 x 3 row vector)
%
%Example: linecolor = colordg(8);
%
%See also: COLOR_ORDERDG.MAT

%Author: clement Etienam
%PhD Supervisor: Dr Rossmary Villegas



color_order = ...
[ 0 0 1 % 1 BLUE
0 0.5 0 % 2 GREEN (medium dark)
1 0 0 % 3 RED
0 0.75 0.75 % 4 TURQUOISE
0.75 0 0.75 % 5 MAGENTA
0.75 0.75 0 % 6 YELLOW (dark)
0.25 0.25 0.25 % 7 GREY (very dark)
1 0.50 0.25 % 8 ORANGE
0.6 0.5 0.4 % 9 BROWN
1 1 0 ]; % 10 YELLOW (pale)

linecolor = color_order(n,:);

%END of code