
format longE;

train_data = importdata('dataset/train/X_train.txt');

data_mean = mean(train_data);

data_cov = cov(train_data);

[evec,eval] = eig(data_cov);

evec = evec(:,size(evec,1):-1:1);
eval = diag(eval);
eval = eval(end:-1:1);

coff1 = top_vecs(eval,evec,train_data,0.90);
coff2 = top_vecs(eval,evec,train_data,-1.05);
