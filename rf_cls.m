clear;

train_data = importdata('dataset/train/X_train.txt');
data_mean = mean(train_data);

train_labels = importdata('dataset/train/y_train.txt');

test_data = importdata('dataset/test/X_test.txt');

test_labels = importdata('dataset/test/y_test.txt');

B = TreeBagger(50,train_data,train_labels,'OOBPred','On','Method','classification');

% oobErrorBaggedEnsemble = oobError(B);
% plot(oobErrorBaggedEnsemble)
% xlabel 'Number of grown trees';
% ylabel 'Out-of-bag classification error';

pred_cls = B.predict(test_data);

cnt  = 0;

for i = 1:2947
    if str2double(pred_cls(i)) == test_labels(i)
        cnt = cnt+1;
    end
end

cnt/2947;

%With PCA Projections
train_proj = pca_cls(train_data,data_mean);

B = TreeBagger(50,train_proj,train_labels,'OOBPred','On','Method','classification');

oobErrorBaggedEnsemble = oobError(B);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

test_proj = pca_cls(test_data,data_mean);

pred_cls = B.predict(test_proj);

cnt  = 0;

for i = 1:2947
    if str2double(pred_cls(i)) == test_labels(i)
        cnt = cnt+1;
    end
end

cnt/2947