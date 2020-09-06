function [y_changed,minc,maxc]=change_scale_y(y,a,b)
% a is the desired lower scale for transformation
%b is the desired upper scale for transformation
up=(y-min(y));
diff=a-b;
down=max(y)-min(y);
value=(up./down).*diff;

y_changed=value+a;
minc=min(y);
maxc=max(y);
end