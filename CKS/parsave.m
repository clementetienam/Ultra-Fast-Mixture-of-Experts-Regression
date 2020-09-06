function parsave(weights_updated,modelNN,Class_all,clfy,Xtrains,ytrains,datass)
save('Regressor.mat', 'weights_updated');
save('Classifier.mat', 'modelNN');
save('clfy.mat', 'clfy');
save('Class_all.mat', 'Class_all');
save('Xtrains.mat', 'Xtrains');
save('ytrains.mat', 'ytrains');
save('Dataused.out','datass','-ascii')
end