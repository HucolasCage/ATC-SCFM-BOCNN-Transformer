%% 贝叶斯优化目标函数
function objective = CNN_BayesOpt(params, p_train, t_train, ps_output, T_train)
    try
        % 构建网络结构
        layers = [
            imageInputLayer([15, 1, 1])
            
            % 第一卷积层
            convolution2dLayer([params.FilterHeight, params.FilterWidth], params.NumFilters1, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer
            
            maxPooling2dLayer([2, 1], 'Stride', [1, 1])
            
            % 第二卷积层
            convolution2dLayer([params.FilterHeight, params.FilterWidth], 2*params.NumFilters1, 'Padding', 'same')
            batchNormalizationLayer
            reluLayer
            
            dropoutLayer(0.1)
            flattenLayer("Name", "flatten")
            fullyConnectedLayer(1)
            regressionLayer];
        
        % 训练选项(固定参数)
        options = trainingOptions('sgdm', ...
            'MiniBatchSize', 32, ...
            'MaxEpochs', 300, ... % 优化时使用较少epoch加速
            'InitialLearnRate', 1e-2, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.1, ...
            'LearnRateDropPeriod', 200, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', false);
        
        % 训练网络
        net = trainNetwork(p_train, t_train, layers, options);
        
        % 预测训练集
        t_sim1 = predict(net, p_train);
        T_sim1 = mapminmax('reverse', t_sim1, ps_output);
        
        % 计算方差作为目标函数
        objective = abs(var(T_sim1')-var(T_train));
    catch
        % 如果训练失败，返回一个大值
        objective = inf;
    end
end