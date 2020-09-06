function [hyper_updated]=optimise_experts_dnn_2(X_train,y_train,Classe)
Xuse=X_train(Classe,:);
yuse=y_train(Classe,:);
input_count = size( Xuse , 2 );
output_count = size( yuse , 2 );

layers = [ ...
    sequenceInputLayer(input_count)
    fullyConnectedLayer(200)
    reluLayer
    fullyConnectedLayer(80)
    reluLayer
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(output_count)
    regressionLayer
    ];

options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'MiniBatchSize', 5 , ...
    'ValidationFrequency',10, ...
    'ValidationPatience',5, ...
    'Verbose',true, ...
    'Plots','training-progress');
if size(Xuse,1)~= 0

hyper_updated = trainNetwork(Xuse',yuse',layers,options);
end

end