clear;

format longE;

train_data = importdata('dataset/train/X_train.txt');

train_labels = importdata('dataset/train/y_train.txt');

test_data = importdata('dataset/test/X_test.txt');

test_labels = importdata('dataset/test/y_test.txt');

data_mean = mean(train_data);
cls_means = zeros(6,size(train_data,2));
cls_sizes = zeros(6,1);

f_sz = size(train_data,2);

sw = zeros(f_sz,f_sz);
sb = zeros(f_sz,f_sz);

for i= min(train_labels):max(train_labels)
    data = train_data(train_labels==i,:);
    cls_means(i,:) = mean(data);
    cls_sizes(i) = size(data,1);
    sw = sw + cov(data);
    sb = sb + (cls_sizes(i) * ((cls_means(i,:) - data_mean)' * (cls_means(i,:) - data_mean)));
end

prod = pinv(sw) * sb;

[evec,eval] = eig(prod);
eval = diag(eval);
eval = real(eval);
evec = real(evec(:,1:5));

cnt = 0;
for i = 1:size(test_data,1)
    t_pr = test_data(i,:) * evec;
    dist = zeros(1,6);
    for j = 1:6
        m_pr = cls_means(j,:) * evec;
        dist(j) = pdist([t_pr;m_pr]);
    end
    [v,ind] = min(dist);
    if(ind == test_labels(i))
        cnt = cnt + 1;
    end
end

cnt/size(test_labels,1)

