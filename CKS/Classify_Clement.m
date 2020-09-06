function hyper=Classify_Clement(X,y,Experts)
idx=y;
idx=categorical(idx);
input_count = size( X , 2 );
output_count = Experts;

layers = [ ...
    sequenceInputLayer(input_count)
    fullyConnectedLayer(200)
    reluLayer
    fullyConnectedLayer(80)
    reluLayer
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(output_count)
    softmaxLayer
    classificationLayer
    ];

options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'MiniBatchSize', 5 , ...
    'ValidationFrequency',10, ...
    'ValidationPatience',5, ...
    'Verbose',true);
hyper = trainNetwork(X',idx',layers,options);
end