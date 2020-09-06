function y_get=undo_scale_y(yin,minc,maxc)
del=maxc-minc;
right=(del*(yin+1))/2;
y_get=minc+ right;

end