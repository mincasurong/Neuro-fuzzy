% fuzzy neural network practice (19.12.23)
clear all; close all; clc;
global number_rules GA_max_gen PSO_max_iter number_MF learning_ratio ...
    Number_Output runtunefis number_out choice SMOTE

%% Initialization
% Switch
runtunefis = 1; % Training // 0: Tunning off 1: Tunning on
% FIS
number_MF = 5; % Max = 5
number_out = 20; % Higher as higher (default: 10)
number_rules = 20; % Higher as higher  (default: 10)

% Optimization (Iteration is more important than the FIS structure)
GA_max_gen = 50;  % First: 2, Final > 20 && < 50
PSO_max_iter = 50; %  First: 2, Final > 20 && < 50
learning_ratio = 0.9;

%% Data implementation
choice = 3;
[data Number_Output] = datachoice(choice);

x = data(:,1:end-Number_Output);     % raw data Input
y = data(:,end-Number_Output+1:end); % raw data Output
rng('default'); shuffle_n = randperm(length(x)); j = 1; h = 1; % Suffle the dataset, and seperate to training and validation set
for k=1:length(x)
    if k < length(x)*learning_ratio
        trnX(j,:) = x(shuffle_n(k),:);
        trnY(j,:) = y(shuffle_n(k),:);
        j = j + 1;
    else
        vldX(h,:) = x(shuffle_n(k),:);
        vldY(h,:) = y(shuffle_n(k),:);
        h = h + 1;
    end
end

% Training and validation set
trnY_eval = zeros(length(trnY),1); vldY_eval = zeros(length(vldY),1);
for k=1:length(trnY), trnY_eval(k,1) = find(trnY(k,:)); end % traninig
for k=1:length(vldY), vldY_eval(k,1) = find(vldY(k,:)); end % Validation

figure('color','w')
histogram(trnY_eval); hold on; histogram(vldY_eval); hold on;
legend('Training','Validation')

%% creat a fis

if runtunefis
    fis = sugfis;
    for k = 1:size(trnX,2)
        fis_range.in = [min(trnX(:,k)) max(trnX(:,k))];
        fis = addInput(fis,fis_range.in,'NumMFs',number_MF,'MFType',"gaussmf");
    end  % Input
    for k=1:size(trnY,2)
        fis_range.out = [0 1];
        numOutputMFs = number_out;
        fis = addOutput(fis,fis_range.out,'NumMFs',numOutputMFs);
    end  % Output
    %% Learn Fuzzy rules
    fisin = fis;
    fisin.Rules = [];
    options1 = tunefisOptions("Method","ga"); % ga, particleswarm, patternsearch, simulannealbnd, anfis
    options1.MethodOptions.MaxGenerations = GA_max_gen;
    options1.OptimizationType = 'learning';
    options1.NumMaxRules = number_rules;
    options1.UseParallel = true;
    options1.DistanceMetric = "norm2"; % default = rmse
    rng('default')  % for reproducibility
    fisout_rule_learning = tunefis(fisin,[],trnX,trnY,options1);
    
    plotfis(fisout_rule_learning)
    gensurf(fisout_rule_learning,gensurfOptions('InputIndex',2))
    %% Tune membership function parameters and fuzzy rules
    % Tuning options
    
    [in,out,rule] = getTunableSettings(fisout_rule_learning);
    options2 = tunefisOptions("Method","particleswarm"); % ga, particleswarm, patternsearch, simulannealbnd, anfis
    options2.MethodOptions.MaxIterations = PSO_max_iter; % MaxIteration MaxGeneration
    options2.UseParallel = true;
    options2.DistanceMetric = "norm2"; % default = rmse
    
    rng('default')  % for reproducibility
    [fisout_final,optimout_final] = tunefis(fisout_rule_learning,[in;out;rule],trnX,trnY,options2);
    
    writeFIS(fisout_final,'myFile');
    
    
else
    fisout_final = readfis('winetest_best');
end
%% Result
opt = evalfisOptions('OutofRangeInputValueMessage','none', ...
    'NoRuleFiredMessage','none','EmptyOutputFuzzySetMessage','none');

% Training set
Final.trn.output = evalfis(fisout_final,trnX,opt);
[Final.trn.output_j, Final.trn.output_softmax] = softmax_custom(trnX,Final.trn.output);

figure('color','w');
plotconfusion(trnY',Final.trn.output_softmax'); title('Training set')

% Validation set
Final.vld.output = evalfis(fisout_final,vldX,opt);
[Final.vld.output_j, Final.vld.output_softmax] = softmax_custom(vldX,Final.vld.output);
figure('color','w'); plotconfusion(vldY',Final.vld.output_softmax'); title('Validation set')


% Graph by samples
for k=1:length(trnY)
    [~,Final.graph(k,1)] = max(Final.trn.output_softmax(k,:));
    [~,Final.graph(k,2)] = max(trnY(k,:));
end
figure('color','w'); 
subplot(211); plot(Final.graph(:,1),'bo'); hold on; plot(Final.graph(:,2),'rx');
axis([0 length(Final.graph) 0 Number_Output+1])
subplot(212); plot(abs(Final.graph(:,1)-Final.graph(:,2)),'bo');
axis([0 length(Final.graph) -1 2])
legend('Predicted','Actual')
%% Figure
ruleview(fisout_final);
% surfview(fisout_final);
figure('color','w'); 
subplot(211); plotmf(fis,'input',1)
subplot(212); plotmf(fisout_final,'input',1)

%% Function

% Softmax
function [Y_j, Y_softmax] = softmax_custom(X,Y)

N = length(X);
Y_softmax = zeros(size(Y));
Y_j = zeros(N,1);
save_softmax = 0;

for k = 1:N
    Y_softmax(k,:) = exp(Y(k,:))/sum(exp(Y(k,:)));
    for j = 1:size(Y,2)
        if Y_softmax(k,j) > save_softmax,
            Y_j(k) = j;
            save_softmax = Y_softmax(k,j);
        end
    end
    save_softmax = 0;
end

end

function [data, Number_Output] = datachoice(X)
switch(X)
    case 1, data = load('wine_test_best.txt'); Number_Output = 3;
    case 2, data = load('data_mimo_clust1_o2.txt'); Number_Output = 2; Input = data(:,1:2); for k=1:length(data), [~,output(k,1)] = max(data(k,3:end)); end; figure; plot3(Input(:,1),Input(:,2),output,'o');
    case 3, data = load('data_mimo_clust2_o4.txt'); Number_Output = 4; Input = data(:,1:2); for k=1:length(data), [~,output(k,1)] = max(data(k,3:end)); end; figure; plot3(Input(:,1),Input(:,2),output,'o');
    case 4, data = load('data_mimo_clust3_o4.txt'); Number_Output = 4; Input = data(:,1:2); for k=1:length(data), [~,output(k,1)] = max(data(k,3:end)); end; figure; plot3(Input(:,1),Input(:,2),output,'o');
    case 5, data = load('data_mimo_clust4_o4.txt'); Number_Output = 4; Input = data(:,1:2); for k=1:length(data), [~,output(k,1)] = max(data(k,3:end)); end; figure; plot3(Input(:,1),Input(:,2),output,'o');
end

end
