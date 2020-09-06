function [Valueehard,Valueesoft]= Plot_perform_dnn(hardtr,...
    softtr,hardts,softts,y,method,folder,...
    X,ind_train,ind_test,oldfolder,Datause,jjm)

Valueehard=zeros(size(X,1),1);
Valueehard(ind_train,:)=hardtr;
Valueehard(ind_test,:)=hardts;
CCR=Valueehard;
True=y;
yesup=sum((abs(CCR-True)).^2);
yesdown=sum((True-mean(True,1)).^2);
R2h=1-(yesup/yesdown);
R2h=R2h*100;


Valueesoft=zeros(size(X,1),1);
Valueesoft(ind_train,:)=softtr;
Valueesoft(ind_test,:)=softts;
CCR=Valueesoft;
True=y;
yesup=sum((abs(CCR-True)).^2);
yesdown=sum((True-mean(True,1)).^2);
R2h1=1-(yesup/yesdown);
R2s=R2h1*100;


%if method= 1 || &&
if (method==1 || 2 || 3 )  && size(X,2)==1   
figure()
subplot(2,3,1)
plot(X,y,'+r');
hold on
plot(X,Valueehard,'.k')
shading flat
grid off
title('(a)-Machine Reconstruction(Hard-Prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Y', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('X', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')
h = legend('True y','Machine');set(h,'FontSize',10);

subplot(2,3,2)
line(Valueehard,y,'Tag','Data','MarkerFaceColor',[1 0 0],...
    'MarkerEdgeColor',[1 0 0],...
    'Marker','o',...
    'LineStyle','none',...
    'Color',[0 0 1]);
title('(b)-Machine Reconstruction(Hard-prediction)','FontName','Helvetica', 'Fontsize', 10);
shading flat
grid off
colormap('jet')
xlabel('Machine','FontSize',10,'FontName','Helvetica');
ylabel('True','FontSize',10,'FontName','Helvetica');     
line([min([Valueehard,y]),max([Valueehard,y])],[min([Valueehard,y]),max([Valueehard,y])],'Tag','Reference Ends','LineWidth',3,'color','black');
str=['R2 = ',num2str(R2h)];
T = text(min(get(gca, 'xlim')), max(get(gca, 'ylim')), str); 
set(T, 'fontsize', 10, 'verticalalignment', 'top', 'horizontalalignment', 'left');

set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')

subplot(2,3,3)
hist(Valueehard-y)
shading flat
grid off
title('(c)-Dissimilarity(Hard-prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Count', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('Difference', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')

subplot(2,3,4)
plot(X,y,'+r');
hold on
plot(X,Valueesoft,'.k')
shading flat
grid off
title('(d)-Machine Reconstruction(Soft-Prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Y', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('X', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')
h = legend('True y','Machine');set(h,'FontSize',10);

subplot(2,3,5)
line(Valueesoft,y,'Tag','Data','MarkerFaceColor',[1 0 0],...
    'MarkerEdgeColor',[1 0 0],...
    'Marker','o',...
    'LineStyle','none',...
    'Color',[0 0 1]);
title('(e)-Machine Reconstruction(Soft-prediction)','FontName','Helvetica', 'Fontsize', 10);
shading flat
grid off
colormap('jet')
xlabel('Machine','FontSize',10,'FontName','Helvetica');
ylabel('True','FontSize',10,'FontName','Helvetica');     
line([min([Valueesoft,y]),max([Valueesoft,y])],[min([Valueesoft,y]),max([Valueesoft,y])],'Tag','Reference Ends','LineWidth',3,'color','black');
str=['R2 = ',num2str(R2s)];
T = text(min(get(gca, 'xlim')), max(get(gca, 'ylim')), str); 
set(T, 'fontsize', 10, 'verticalalignment', 'top', 'horizontalalignment', 'left');

set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')
subplot(2,3,6)
hist(Valueesoft-y)
shading flat
grid off
title('(f)-Dissimilarity(Soft-prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Count', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('Difference', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')
cd(folder)
Namefilef=strcat('performance_', sprintf('%d',jjm),'.fig');
saveas(gcf,Namefilef)
cd(oldfolder)
end

if (method==1 || 2 || 3 )  && (size(X,2)>=2)
    
figure()
subplot(2,2,1)
line(Valueehard,y,'Tag','Data','MarkerFaceColor',[1 0 0],...
    'MarkerEdgeColor',[1 0 0],...
    'Marker','o',...
    'LineStyle','none',...
    'Color',[0 0 1]);
title('(a)-Machine Reconstruction(Hard-prediction)','FontName','Helvetica', 'Fontsize', 10);
shading flat
grid off
colormap('jet')
xlabel('Machine','FontSize',10,'FontName','Helvetica');
ylabel('True','FontSize',10,'FontName','Helvetica');     
line([min([Valueehard,y]),max([Valueehard,y])],[min([Valueehard,y]),max([Valueehard,y])],'Tag','Reference Ends','LineWidth',3,'color','black');
str=['R2 = ',num2str(R2h)];
T = text(min(get(gca, 'xlim')), max(get(gca, 'ylim')), str); 
set(T, 'fontsize', 10, 'verticalalignment', 'top', 'horizontalalignment', 'left');

set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')

subplot(2,2,2)
hist(Valueehard-y)
shading flat
grid off
title('(b)-Dissimilarity(Hard-prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Count', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('Difference', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')

subplot(2,2,3)
line(Valueesoft,y,'Tag','Data','MarkerFaceColor',[1 0 0],...
    'MarkerEdgeColor',[1 0 0],...
    'Marker','o',...
    'LineStyle','none',...
    'Color',[0 0 1]);
title('(c)-Machine Reconstruction(Soft-prediction)','FontName','Helvetica', 'Fontsize', 10);
shading flat
grid off
colormap('jet')
xlabel('Machine','FontSize',10,'FontName','Helvetica');
ylabel('True','FontSize',10,'FontName','Helvetica');     
line([min([Valueesoft,y]),max([Valueesoft,y])],[min([Valueesoft,y]),max([Valueesoft,y])],'Tag','Reference Ends','LineWidth',3,'color','black');
str=['R2 = ',num2str(R2s)];
T = text(min(get(gca, 'xlim')), max(get(gca, 'ylim')), str); 
set(T, 'fontsize', 10, 'verticalalignment', 'top', 'horizontalalignment', 'left');

set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')

subplot(2,2,4)
hist(Valueesoft-y)
shading flat
grid off
title('(d)-Dissimilarity(Soft-prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Count', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('Difference', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')
cd(folder)
Namefilef=strcat('performance_', sprintf('%d',jjm),'.fig');
saveas(gcf,Namefilef)
cd(oldfolder)
end

end