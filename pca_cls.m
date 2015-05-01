function [coff] = pca_cls(train_data,data_mean)

    data_cov = cov(train_data);

    [evec,eval] = eig(data_cov);

    evec = evec(:,size(evec,1):-1:1);
    eval = diag(eval);
    eval = eval(end:-1:1);

    coff = top_vecs(eval,evec,train_data,data_mean,0.90);

    % plot(coff1(:,1),coff1(:,2),'k*','MarkerSize',5);
    % plot(coff2(:,1),coff2(:,2),'k*','MarkerSize',5);
end