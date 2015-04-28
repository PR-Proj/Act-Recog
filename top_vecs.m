%returns principal component coffecients

function [coff] = top_vecs(eval,evec,data,req_perc)
    perc = 0;
    total = sum(eval);
    temp = 0;
    
    data_mean = mean(data);
    
    for i = 1:size(eval)
        temp = temp + eval(i);
        perc = temp/total;
        if perc > req_perc
            break;
        end
    end
    i = i-1;

    top_eigen = eval(1:i);
    top_evec = evec(:,1:i);

    top_ei_perc = top_eigen/sum(top_eigen);

    % EigenSpectrum Bar Plot
    bar(top_ei_perc);
    
    cent_data = (data - repmat(data_mean,size(data,1),1));
    coff = top_evec' * cent_data';
    
end