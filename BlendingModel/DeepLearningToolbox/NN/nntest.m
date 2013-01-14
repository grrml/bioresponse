function [er, bad, prob] = nntest(nn, x, y)
    nn.testing = 1;
    nn = nnff(nn, x, y);
    nn.testing = 0;
    %xlswrite('probdbn.csv',nn.a{end});
    prob = nn.a{end}(:,1);
    [~, i] = max(nn.a{end},[],2);
    [~, g] = max(y,[],2);
    bad = find(i ~= g);    
    %xlswrite('preddbn.csv',g);
    %xlswrite('pred1dbn.csv',i);
    er = numel(bad) / size(x, 1);
end
