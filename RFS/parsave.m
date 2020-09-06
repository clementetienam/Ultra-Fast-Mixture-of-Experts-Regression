function parsave(weights_updated,modelNN,Class_all,clfy,clfx,datass)
save('Regressor.mat', 'weights_updated');
save('Classifier.mat', 'modelNN');
save('clfy.mat', 'clfy');
save('clfx.mat', 'clfx');
save('Class_all.mat', 'Class_all');
save('Dataused.out','datass','-ascii')
end